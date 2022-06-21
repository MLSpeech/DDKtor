__author__ = 'YaelSegal'
import numpy as np
import torch
import torch.nn as nn
import os
import numpy as np
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helpers import textgrid
SIL=0
VOT=1
VOWEL=2
SR = 16000

def get_type(name):
    if "vowel" in name:
        return VOWEL
    if "vot" in name:
        return VOT
    return SIL

def get_name_by_type(ftype):
    if ftype == VOWEL:
        return "Vowel"
    if ftype == VOT:
        return "Vot"
    return ""
# Felix code!
class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd
    def forward(self, x):
        return self.lambd(x)


def pad_tensor(vec, pad, dim):
    """
    args:
        vec - tensor to pad
        pad - the size to pad to
        dim - dimension to pad
    return:
        a new tensor padded to 'pad' in dimension 'dim'
    """

    pad_size = list(vec.shape)
    pad_size[dim] = pad - vec.size(dim)
    return torch.cat([vec, torch.zeros(*pad_size).to(vec.device)], dim=dim)

def padd_list_tensors(targets, targets_lens, dim):

    target_max_len = max(targets_lens)
    padded_tensors_list = []
    for tensor, tensor_len in zip(targets, targets_lens):
        pad = pad_tensor(tensor, target_max_len, dim)
        padded_tensors_list.append(pad)
    padded_tensors = torch.stack(padded_tensors_list)
    return padded_tensors

def merge_close(sections_list):

    merge_section = [sections_list[0]]
    for index in range(1,len(sections_list)):
        prev_item = merge_section.pop()
        current_item = sections_list[index]
        if prev_item[2] == current_item[2]:
            merge_section.append([prev_item[0], current_item[1], current_item[2], current_item[1]-prev_item[0]])
        else:
            merge_section.append(prev_item)
            merge_section.append(current_item)

    return merge_section

def merge_type(sections_list, ftype, gap=20):
    if len(sections_list)< 2:
        return sections_list
    merge_section = [sections_list[0], sections_list[1]]

    for index in range(2,len(sections_list)):

        if len(merge_section)<2:

            middle_item = merge_section.pop()
            last_item = sections_list[index]
            merge_section.append(middle_item)
            merge_section.append(last_item)
            continue
        else:
            middle_item = merge_section.pop()
            first_item = merge_section.pop()
  
        last_item = sections_list[index]
        if  middle_item[3]<gap and middle_item[2]==SIL and last_item[2]==ftype and first_item[2]==ftype:
            merge_section.append([first_item[0],last_item[1], ftype, first_item[3]+middle_item[3]+last_item[3]])
        else:
            merge_section.append(first_item)
            merge_section.append(middle_item)
            merge_section.append(last_item)

    return merge_section

def process_sections(preds_array, pre_process=False):
    change_value = np.diff(preds_array) 
    change_value_idx =  np.argwhere(change_value != 0)
    sections_list = []
    start_idx = 0
    for idx in change_value_idx:
        idx = idx[0]
        mark = preds_array[idx]
        item_len = idx - start_idx +1
        remove = False
        if get_name_by_type(mark) == get_name_by_type(VOT) and item_len<5 \
            or get_name_by_type(mark) == get_name_by_type(VOWEL) and item_len <20:
            remove = True
 
        sections_list.append([start_idx, idx+1,mark, item_len, remove])
        start_idx = idx+1
    if start_idx != len(preds_array):
        sections_list.append([start_idx, len(preds_array)-1, preds_array[-1], len(preds_array) - start_idx, False])
    if pre_process:
        new_sections_list = []
        for idx, (start_idx, end_idx, mark, item_len,remove) in enumerate(sections_list):
            if not remove:
                new_sections_list.append([start_idx, end_idx, mark, item_len])
                continue
            prev_item = sections_list[idx-1] if idx-1 >= 0 else None
            new_sections_list.append([start_idx, end_idx, SIL, item_len])
            if prev_item and prev_item[2] == SIL:
                new_sections_list.pop()
                new_sections_list.append([prev_item[0], end_idx, SIL, end_idx- prev_item[0]])
        new_sections_list = merge_close(new_sections_list)
        new_sections_list = merge_type(new_sections_list, VOT)

        return new_sections_list
    else:
        return [x[:-1] for x in sections_list]

def create_textgrid(preds_array, new_filename, wav_len):

    sections_list = process_sections(preds_array, True)
    new_textgrid = textgrid.TextGrid()
    tier = textgrid.IntervalTier(name="preds", minTime=0)

    for item in sections_list:
        start_item, end_item, mark, item_len = item
        start_sec = start_item / 1000
        end_sec = end_item / 1000
        add = ""
        if get_name_by_type(mark) == get_name_by_type(VOT) and item_len <5:
            add = "short {}".format(item_len)
            print("file:{},vot {}:{}".format(new_filename, start_sec, end_sec))
        elif get_name_by_type(mark) == get_name_by_type(VOWEL) and item_len <30:
            add = "short {}".format(item_len)
            print("file:{},short vowel {}:{}".format(new_filename, start_sec, end_sec))

        tier.add(start_sec, end_sec, get_name_by_type(mark) + add)


    new_textgrid.append(tier)
    new_textgrid.write(new_filename)


def fix_extention(path):
    if path.endswith(".WAV"):
        os.system("mv {} {}".format(path, path.replace(".WAV", ".wav")))