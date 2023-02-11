from collections import defaultdict

import numpy as np
import pytorch_lightning as pl
import torch_optimizer as optim
import torch
from torch import Tensor, nn
import torch.nn.functional as F
import math
from tqdm import tqdm
from models.diff_decoder import MIDI2SpecDiff
from .mel import MelFeature
from .scaler import get_scaler
from .eval_utils import (get_models, calculate_metrics, aggregate_metrics, get_wav, 
                         StreamingMultivariateGaussian)


def log_snr2snr(log_snr: Tensor) -> Tensor:
    return torch.exp(log_snr)


def snr2as(snr: Tensor):
    snr_p1 = snr + 1
    return torch.sqrt(snr / snr_p1), snr_p1.reciprocal()


def log_snr2as(log_snr: Tensor):
    var = (-log_snr).sigmoid()
    return (1 - var).sqrt(), var


def log_snr2logas(log_snr: Tensor):
    log_var = -F.softplus(log_snr)
    return 0.5 * (log_snr + log_var), log_var


class DiffusionLM(pl.LightningModule):
    logsnr_min = -20
    logsnr_max = 20

    def __init__(self,
                 num_emb: int = 900,
                 output_dim: int = 128,
                 max_input_length: int = 2048,
                 max_output_length: int = 512,
                 emb_dim: int = 512,
                 dim_feedforward: int = 1024,
                 nhead: int = 6,
                 head_dim: int = 64,
                 num_layers: int = 8,
                 cfg_dropout: float = 0.1,
                 cfg_weighting: float = 2.0,
                 with_context: bool = False,
                 dropout: float = 0.1,
                 layer_norm_eps: float = 1e-5,
                 norm_first: bool = True,
                 **mel_kwargs) -> None:
        super().__init__()

        self.cfg_dropout = cfg_dropout
        self.cfg_weighting = cfg_weighting
        self.output_dim = output_dim

        self.model = MIDI2SpecDiff(
            num_emb=num_emb, output_dim=output_dim, max_input_length=max_input_length,
            max_output_length=max_output_length, emb_dim=emb_dim, nhead=nhead, with_context=with_context,
            head_dim=head_dim, num_encoder_layers=num_layers, num_decoder_layers=num_layers, dropout=dropout,
            layer_norm_eps=layer_norm_eps, norm_first=norm_first, dim_feedforward=dim_feedforward
        )

        self.mel = MelFeature(window_fn=torch.hann_window, **mel_kwargs)

    def get_log_snr(self, t):
        """Compute Cosine log SNR for a given time step."""
        b = math.atan(math.exp(-0.5 * self.logsnr_max))
        a = math.atan(math.exp(-0.5 * self.logsnr_min)) - b
        return -2.0 * torch.log(torch.tan(a * t + b))

    def get_training_inputs(self, x: torch.Tensor, uniform: bool = False, minibatch_sync=False):
        N = x.shape[0]
        if uniform:
            t = torch.linspace(0, 1, N).to(x.device)
        else:
            t = x.new_empty(N).uniform_(0, 1)
        if minibatch_sync:  # this setting is only for return a random but same t for all samples in a minibatch
            t = t[0] * torch.ones_like(t)
        log_snr = self.get_log_snr(t)
        alpha, var = log_snr2as(log_snr)
        sigma = var.sqrt()
        noise = torch.randn_like(x)
        z_t = x * alpha[:, None, None] + sigma[:, None, None] * noise
        return z_t, t, noise

    def forward(self, batch_size=4, seq_length=256, mel_context=None,
                wav_context=None, rescale=True, T=1000, verbose=True,
                fast_sampling=False, fast_sampling_schedule=None,
                inject_grad_model=None, text_input=None,
                stop_t=-1):
        if wav_context is not None:
            context = self.mel(wav_context)
        elif mel_context is not None:
            context = mel_context
        else:
            context = None

        t = torch.linspace(0, 1, T).to(self.device)  # t from 0 to 1
        log_snr = self.get_log_snr(t)  # from 20 to -20
        log_alpha, log_var = log_snr2logas(log_snr)

        var = log_var.exp()  # noise schedule, increasing, from 0 to 1
        alpha = log_alpha.exp()  # should be decreasing, it is cum alpha, from 1 to 0
        alpha_st = torch.exp(log_alpha[:-1] - log_alpha[1:])  # a trick to calculate {1 / alpha_t}
        c = -torch.expm1(log_snr[1:] - log_snr[:-1])
        c.relu_()

        # z_t start from a random noise.
        z_t = torch.randn(
            batch_size, seq_length, self.output_dim, device=self.device)

        dropout_mask = torch.tensor(
            [0] * batch_size + [1] * batch_size).bool().to(self.device)
        t = torch.broadcast_to(t, (batch_size * 2, T))
        if context is not None:
            context = context.repeat(2, 1, 1)

        for t_idx in tqdm(range(T - 1, -1, -1), disable=True):
            s_idx = t_idx - 1

            # if fast_sampling and t_idx not in T_fast and s_idx > 0:
            #     continue

            # calculate the noise prediction
            noise_hat = self.model(z_t.repeat(
                2, 1, 1), t[:, t_idx], context, dropout_mask=dropout_mask)
            cond_noise_hat, uncond_noise_hat = noise_hat.chunk(2, dim=0)
            noise_hat = cond_noise_hat * self.cfg_weighting + uncond_noise_hat * (1 - self.cfg_weighting)
            # noise_hat = uncond_noise_hat  # we don't use cond noise

            if inject_grad_model: # CLAP guidance
                zt_input = z_t.clone()
                zt_input.requires_grad = True
                loss_grad = inject_grad_model.get_gradient_to_input(zt_input, text_input, t_idx / T)
                grad = torch.autograd.grad(loss_grad, zt_input, allow_unused=True)[0]
                noise_hat = noise_hat + 16 * grad
                
            noise_hat = noise_hat.clamp(
                (z_t - alpha[t_idx]) * var[t_idx].rsqrt(),  # rsqrt = 1 / sqrt
                (alpha[t_idx] + z_t) * var[t_idx].rsqrt(),
            )

            if s_idx >= 0:
                mu = (z_t - var[t_idx].sqrt() * c[s_idx]
                      * noise_hat) * alpha_st[s_idx]
                z_t = mu + (var[s_idx] * c[s_idx]).sqrt() * torch.randn_like(z_t)
                continue

            final = (z_t - var[0].sqrt() * noise_hat) / alpha[0]
        if rescale:
            final = self.mel[1].reverse(final)

        return final


    def training_step(self, batch, batch_idx):
        wav, *_ = batch
        spec = self.mel(wav)
        if len(_) > 0:
            context = _[0]
            context = self.mel(context)
        else:
            context = None
        N = wav.shape[0]
        dropout_mask = spec.new_empty(N).bernoulli_(self.cfg_dropout).bool()
        z_t, t, noise = self.get_training_inputs(spec)
        noise_hat = self.model(z_t, t, context,
                               dropout_mask=dropout_mask)
        loss = F.l1_loss(noise_hat, noise)

        values = {
            'loss': loss,
        }
        self.log_dict(values, prog_bar=False, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        wav, *_ = batch
        spec = self.mel(wav)
        if len(_) > 0:
            context = _[0]
            context = self.mel(context)
        else:
            context = None
        z_t, t, noise = self.get_training_inputs(spec, uniform=True)
        noise_hat = self.model(z_t, t, context)
        loss = F.l1_loss(noise_hat, noise)
        return loss.item()

    def validation_epoch_end(self, outputs) -> None:
        avg_loss = sum(outputs) / len(outputs)
        self.log('val_loss', avg_loss, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        return optim.Adafactor(self.parameters(), lr=1e-3)

    def on_test_start(self) -> None:
        vggish_model, trill_model, self.melgan = get_models()
        self.vggish_fn = lambda x, sr: vggish_model(x)
        self.trill_fn = lambda x, sr: trill_model(
            x, sample_rate=sr)['embedding']
        
        self.true_dists = defaultdict(StreamingMultivariateGaussian)
        self.pred_dists = defaultdict(StreamingMultivariateGaussian)

        return super().on_test_start()

    def test_step(self, batch, batch_idx):
        orig_wav, *_ = batch
        spec = self.mel(orig_wav)
        if len(_) > 0:
            context = _[0]
            context = self.mel(context)
        else:
            context = None
        z_t, t, noise = self.get_training_inputs(spec, uniform=True)
        noise_hat = self.model(z_t, t, context)
        loss = F.l1_loss(noise_hat, noise)
        pred = self.forward(batch_size=spec.shape[0],
            seq_length=spec.shape[1], mel_context=context, verbose=False)
        pred_wav = self.spec_to_wav(pred)
        orig_wav = orig_wav.cpu().numpy()
        metric = calculate_metrics(
            orig_wav, pred_wav, self.vggish_fn, self.trill_fn, self.true_dists, self.pred_dists
        )
        metric["loss"] = loss.item()
        return metric

    def test_epoch_end(self, outputs) -> None:
        metrics = aggregate_metrics(outputs, self.true_dists, self.pred_dists)
        self.log_dict(metrics, prog_bar=True, sync_dist=True)

        return super().test_epoch_end(outputs)

    def spec_to_wav(self, spec):
        return get_wav(self.melgan, spec)
