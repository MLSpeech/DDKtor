__author__ = 'YaelSegal'
import os.path
import random
import librosa
import numpy as np
import torch
import torch.utils.data as data
import soundfile
import random
import glob
import utils
from utils import SIL, VOT,VOWEL, SR




def make_dataset(data_path, slices_size, overlap, predict):

    files_list = glob.glob(data_path + "/*.wav")
    wav_labels_dict = {}
    dataset = []
    for wav_filename in files_list:
        y, sr = soundfile.read(wav_filename)
        wav_duration = len(y)/SR

        phn_filename = wav_filename.replace(".wav", ".phn")
        file_labels_list = open(phn_filename, "r").readlines()
        # 3 classes - sil + all , vot, vowel
        labels_array = np.zeros(int(wav_duration*1000))
        labels_list = []
        for line in file_labels_list:
            line_array = line.strip().split(" ")
            start_frame = int(line_array[0])
            end_frame = int(line_array[1])
            start_idx = int(start_frame / 16)
            end_idx = int(end_frame / 16)
            frame_type= utils.get_type(line_array[2])
            labels_array[start_idx:end_idx] = frame_type
            labels_list.append([start_idx, end_idx,frame_type,line_array[2] ])

        start = 0
        if predict:
            dataset.append([y, start, labels_array,wav_filename])
        else:
            
            while start < len(y) - overlap:
                end = min(start + slices_size * 16, len(y))
                start_ms = int(start/16)
                end_ms = int(end/16)
                dataset.append([y[start: end], start, labels_array[start_ms: end_ms],wav_filename])
                start = end - overlap
        
        wav_labels_dict[wav_filename] = labels_list
    
    return dataset, wav_labels_dict


class PredictDataset(data.Dataset):
    def __init__(self, data_path, seed, slices_size=250, overlap=40, normalize=True, norm_type='z'):
        np.random.seed(seed)
        random.seed(seed)
        self.norm_type = norm_type
        self.normalize = normalize
        y, sr = soundfile.read(data_path)
        if sr !=SR:
            print("sample rate not compatible, sr: {}, should be: {}".format(sr, SR))
            new_y = librosa.resample(y,sr,SR)
            y = new_y

        self.wav_duration = len(y)/SR
        dataset = []
        start = 0
        while start < len(y):
            end = min(start + slices_size * 16, len(y))
            if end- start < 32:
                break
            start_ms = int(start/16)
            end_ms = int(end/16)
            current_len = end_ms - start_ms
            dataset.append([y[start: end], current_len])
            start = end-overlap if end!=len(y) else end

        self.dataset = dataset
        self.slices_size = slices_size
        self.overlap= overlap
        self.normalize = normalize


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (spect, target) where target is class_index of the target class.
        """
        y, seq_len = self.dataset[index]
        if self.normalize:
            if self.norm_type == 'z':
                mean_y, std_y = y.mean(), y.std()
                y -= mean_y
                y /= std_y
            elif self.norm_type == 'minmax':
                a, b = -1, 1
                new_minmax_y = np.copy(y.numpy())
                y = a + ((new_minmax_y - min(new_minmax_y))*(b-a))/(max(new_minmax_y) - min(new_minmax_y))
                y = torch.FloatTensor(y)
            else:
                y -=y.mean()

        y_tensor = torch.FloatTensor(y)

        return y_tensor, seq_len


    def __len__(self):
        return len(self.dataset)

class PadCollatePred:
    """
    a variant of callate_fn that pads according to the longest sequence in
    a batch of sequences
    """

    def __init__(self, dim=0):
        """
        args:
            dim - the dimension to be padded (dimension of time in sequences)
        """
        self.dim = dim

    def pad_collate(self, batch):
        """
        args:
            batch - y_tensor, labels_tensor, len(labels_array)
        reutrn:
            xs - a tensor of all examples in 'batch' after padding
            ys - a LongTensor of all labels in batch
        """
        # find longest sequence
        wav = [x[0] for x in batch]
        wav_len = [x[0].size(0) for x in batch]
        wav = utils.padd_list_tensors(wav, wav_len, dim=self.dim)

        target_len = [x[1] for x in batch]

        return wav, target_len



    def __call__(self, batch):
        return self.pad_collate(batch)


class PadCollateRaw:
    """
    a variant of callate_fn that pads according to the longest sequence in
    a batch of sequences
    """

    def __init__(self, dim=0):
        """
        args:
            dim - the dimension to be padded (dimension of time in sequences)
        """
        self.dim = dim

    def pad_collate(self, batch):
        """
        args:
            batch - y_tensor, labels_tensor, len(labels_array)
        reutrn:
            xs - a tensor of all examples in 'batch' after padding
            ys - a LongTensor of all labels in batch
        """
        # find longest sequence
        wav = [x[0] for x in batch]
        wav_len = [x[0].size(0) for x in batch]
        wav = utils.padd_list_tensors(wav, wav_len, dim=self.dim)

        target = [x[1] for x in batch]
        target_len = [x[2] for x in batch]
        target = utils.padd_list_tensors(target, target_len, self.dim)

        return wav, target, target_len



    def __call__(self, batch):
        return self.pad_collate(batch)
