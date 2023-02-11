import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, ConcatDataset, default_collate, WeightedRandomSampler
from torch.nn.utils.rnn import pad_sequence
from itertools import chain
from data.music4all import Music4All
from preprocessor.event_codec import Codec
import pickle
import os


def get_padding_collate_fn(output_size: int):
    def collate_fn(batch):
        """Pad the batch to the longest sequence."""
        seqs = [item[0] for item in batch]
        rest = [item[1:] for item in batch]
        rest = default_collate(rest)
        if output_size is not None:
            seqs = [torch.cat([seq, seq.new_zeros(output_size - len(seq))])
                    if len(seq) < output_size else seq[:output_size] for seq in seqs]
            seqs = torch.stack(seqs, dim=0)
        else:
            seqs = pad_sequence(seqs, batch_first=True, padding_value=0)
        return seqs, *rest

    return collate_fn


def load_dataset(path):
    return pickle.load(open(path, 'rb'))


def save_dataset(dataset, path):
    pickle.dump(dataset, open(path, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)


class ConcatData(pl.LightningDataModule):
    def __init__(self,
                 batch_size: int = 32,
                 midi_output_size: int = 2048,
                 with_context: bool = True,
                 sample_rate: int = 16000,
                 segment_length: int = 81920,
                 music4all_path: str = "/import/c4dm-04/yz007/music4all/",
                 sampling_temperature: float = 0.3,
                 cache=False,
                 cache_folder="/import/c4dm-04/yz007/"
                 ):
        super().__init__()
        self.save_hyperparameters()
        [self.batch_size, self.midi_output_size, self.with_context, self.sample_rate, self.segment_length,
         self.music4all_path, self.sampling_temperature] = batch_size, midi_output_size, with_context, sample_rate, \
            segment_length, music4all_path, sampling_temperature
        self.cache = cache
        self.cache_folder = cache_folder

    def setup(self, stage=None):
        resolution = 100
        segment_length_in_time = self.segment_length / self.sample_rate
        codec = Codec(int(segment_length_in_time * resolution + 1))

        factory_kwargs = {
            'codec': codec,
            'resolution': resolution,
            'sample_rate': self.sample_rate,
            'segment_length': self.segment_length,
            'with_context': self.with_context,
        }

        if stage == "fit":

            if self.cache and os.path.exists(f"{self.cache_folder}/fit.pkl"):
                self.train_dataset = load_dataset(f"{self.cache_folder}/fit.pkl")
            else:
                train_datasets = []
                if self.music4all_path is not None:
                    train_datasets.append(
                        Music4All(path=self.music4all_path, split='train', **factory_kwargs))

                self.train_dataset = ConcatDataset(train_datasets)
                if self.cache:
                    save_dataset(self.train_dataset, f"{self.cache_folder}/fit.pkl")

            train_num_samples = [len(dataset) for dataset in self.train_dataset.datasets]
            dataset_weights = [
                x ** self.sampling_temperature for x in train_num_samples]
            print("Train dataset sizes: ", train_num_samples)
            print("Train dataset weights: ", dataset_weights)
            self.sampler_weights = list(
                chain.from_iterable(
                    [dataset_weights[i] / train_num_samples[i]] * train_num_samples[i] for i in
                    range(len(train_num_samples))
                )
            )

        if stage == "validate" or stage == "fit":
            if self.cache and os.path.exists(f"{self.cache_folder}/validate.pkl"):
                self.val_dataset = load_dataset(f"{self.cache_folder}/validate.pkl")
            else:
                val_datasets = []
                if self.music4all_path is not None:
                    val_datasets.append(
                        Music4All(path=self.music4all_path, split='val', **factory_kwargs))

                self.val_dataset = ConcatDataset(val_datasets)
                if self.cache:
                    save_dataset(self.val_dataset, f"{self.cache_folder}/validate.pkl")


        if stage == "test":
            if self.cache and os.path.exists(f"{self.cache_folder}/test.pkl"):
                self.test_dataset = load_dataset(f"{self.cache_folder}/test.pkl")
            else:
                test_datasets = []
                if self.music4all_path is not None:
                    test_datasets.append(
                        Music4All(path=self.music4all_path, split='test', **factory_kwargs))

                self.test_dataset = ConcatDataset(test_datasets)
                if self.cache:
                    save_dataset(self.test_dataset, f"{self.cache_folder}/test.pkl")

    def train_dataloader(self):
        # collate_fn = get_padding_collate_fn(self.midi_output_size)
        sampler = WeightedRandomSampler(self.sampler_weights, len(
            self.sampler_weights), replacement=True)
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          sampler=sampler,
                          shuffle=False, num_workers=2)  # , collate_fn=collate_fn)

    def val_dataloader(self):
        # collate_fn = get_padding_collate_fn(self.midi_output_size)
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=2)  # , collate_fn=collate_fn)

    def test_dataloader(self):
        # collate_fn = get_padding_collate_fn(self.midi_output_size)
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=2)  # , collate_fn=collate_fn)
