import io
import av
import os
from torch.utils.data import Dataset as TorchDataset, ConcatDataset
import numpy as np
import h5py
import torch

from fairseq.data import FairseqDataset


def get_dataset(data_path: str, split: str, sample_rate: int = 32000):
    if split == 'train':
        balanced_train_hdf5 = os.path.join(data_path, "balanced_train_segments_mp3.hdf")
        unbalanced_train_hdf5 = os.path.join(data_path, "unbalanced_train_segments_mp3.hdf")
        sets = [AudioSetDataset(balanced_train_hdf5, sample_rate=sample_rate),
                AudioSetDataset(unbalanced_train_hdf5, sample_rate=sample_rate)]
        ds = ConcatDataset(sets)
    elif split == 'valid' or split == 'test' or split == 'eval':
        eval_hdf5 = os.path.join(data_path, "eval_segments_mp3.hdf")
        ds = AudioSetDataset(eval_hdf5, sample_rate=sample_rate)
    else:
        raise ValueError(f"Split '{split}' not implemented. Must be in ['train', 'valid', 'test'].")
    return FairseqAudioSetDataset(ds)


class FairseqAudioSetDataset(FairseqDataset):
    def collater(self, samples):
        assert len(samples) > 0
        wavs = torch.stack([torch.from_numpy(s[0]) for s in samples])
        labels = torch.stack([torch.from_numpy(s[2]) for s in samples])
        batch = {
            "net_input": {"audio": wavs, "labels": labels}
        }
        return batch

    def num_tokens(self, index):
        return 1

    def size(self, index):
        return 1

    def __init__(self,
                 dataset,
                 shuffle=True
                 ):
        self.dataset = dataset
        self.shuffle = shuffle

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        # return len(self.dataset)
        return 1000

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            order = [np.random.permutation(len(self))]
        else:
            order = [np.arange(len(self))]

        return order[0]


class AudioSetDataset(TorchDataset):
    def __init__(self, hdf5_file, sample_rate=32000, classes_num=527, clip_length=10):
        """
        Reads the mp3 bytes from HDF file decodes using av and returns a fixed length audio wav
        """
        self.sample_rate = sample_rate
        self.hdf5_file = hdf5_file
        with h5py.File(hdf5_file, 'r') as f:
            self.length = len(f['audio_name'])
            print(f"Dataset from {hdf5_file} with length {self.length}.")
        self.dataset_file = None  # lazy init
        self.clip_length = clip_length * sample_rate
        self.classes_num = classes_num

    def open_hdf5(self):
        self.dataset_file = h5py.File(self.hdf5_file, 'r')

    def __len__(self):
        return self.length

    def __del__(self):
        if self.dataset_file is not None:
            self.dataset_file.close()
            self.dataset_file = None

    def __getitem__(self, index):
        """Load waveform and target of an audio clip.
        Args:
          meta: {
            'hdf5_path': str,
            'index_in_hdf5': int}
        Returns:
          data_dict: {
            'audio_name': str,
            'waveform': (clip_samples,),
            'target': (classes_num,)}
        """
        # for debugging
        # return np.ones(320000), np.ones(527), np.ones(527)

        if self.dataset_file is None:
            self.open_hdf5()

        audio_name = self.dataset_file['audio_name'][index].decode()
        waveform = decode_mp3(self.dataset_file['mp3'][index])
        waveform = pad_or_truncate(waveform, self.clip_length)
        waveform = self.resample(waveform)
        target = self.dataset_file['target'][index]
        target = np.unpackbits(target, axis=-1,
                               count=self.classes_num).astype(np.float32)

        return waveform, audio_name, target

    def resample(self, waveform):
        """Resample.
        Args:
          waveform: (clip_samples,)
        Returns:
          (resampled_clip_samples,)
        """
        if self.sample_rate == 32000:
            return waveform
        elif self.sample_rate == 16000:
            return waveform[0:: 2]
        elif self.sample_rate == 8000:
            return waveform[0:: 4]
        else:
            raise Exception('Incorrect sample rate!')


def decode_mp3(mp3_arr):
    """
    decodes an array if uint8 representing an mp3 file
    :rtype: np.array
    """
    container = av.open(io.BytesIO(mp3_arr.tobytes()))
    stream = next(s for s in container.streams if s.type == 'audio')
    # print(stream)
    a = []
    for i, packet in enumerate(container.demux(stream)):
        for frame in packet.decode():
            a.append(frame.to_ndarray().reshape(-1))
    waveform = np.concatenate(a)
    if waveform.dtype != 'float32':
        raise RuntimeError("Unexpected wave type")
    return waveform


def pad_or_truncate(x, audio_length):
    """Pad all audio to specific length."""
    if len(x) <= audio_length:
        return np.concatenate((x, np.zeros(audio_length - len(x), dtype=np.float32)), axis=0)
    else:
        return x[0: audio_length]
