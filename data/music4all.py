import os
from tqdm import tqdm
import soundfile as sf

from .common import Base

class Music4All(Base):
    def __init__(self,
                 path: str = "/import/c4dm-04/yz007/music4all/",
                 split: str = "train",
                 **kwargs):
        data_list = []
        file_names = os.listdir(f"{path}/audios_wav/")
        # split list by 8:1:1
        train_files, val_files, test_files = file_names[:int(len(file_names)*0.8)], \
            file_names[int(len(file_names)*0.8):int(len(file_names)*0.9)], \
            file_names[int(len(file_names)*0.9):]
        if split == "train":
            file_names = train_files
        elif split == "val":
            file_names = val_files
        elif split == "test":
            file_names = test_files
        else:
            raise ValueError(f'Invalid split: {split}')
        
        for file in tqdm(file_names):
            title = file.split(".")[0]

            wav_file = f"{path}/audios_wav/{title}.wav"
            info = sf.info(wav_file)
            sr = info.samplerate
            frames = info.frames
            data_list.append((wav_file, sr, frames))

        super().__init__(data_list, **kwargs)
