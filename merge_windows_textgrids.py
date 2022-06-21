__author__ = 'YaelSegal'

import argparse
import os
from helpers.textgrid import *
from helpers.utilities import basic_hierarchy_dict,get_hierarchy_path
import glob

parser = argparse.ArgumentParser(description='copy all wav files from all sub dirs to out_dir')
parser.add_argument('--input_dir', type=str, help='Path of wavs dir',default="./data/out_tg/tmp_parts")
parser.add_argument('--output_dir', type=str, help='Path to output dir',default="./data/out_tg")
parser.add_argument('--pred_tier',  type=str, default="preds", help='pred tier in textgrid')
parser.add_argument('--basic_hierarchy_file', default="./data/raw/all_files/files.txt", type=str, help="features dir")
parser.add_argument('--durations', default="./data/raw/all_files/voice_starts.txt", type=str, help="start for part files")
parser.add_argument('--use_prev_textgrid', action='store_true', help='use prev textgrid')


def duration2dict(duration_file):
    files_start_list = open(duration_file, "r").readlines()
    durations_dict = {}
    for file_line in files_start_list:
        wav_file, voice_starts, wav_duration = file_line.strip().split(",")
        basename = wav_file.split("/")[-1]
        voice_starts, wav_duration = float(voice_starts), float(wav_duration)
        durations_dict[basename] = [max(voice_starts, 0), wav_duration]
    return durations_dict

args = parser.parse_args()
try:
    if args.basic_hierarchy_file:   
        hierarchy_dict = basic_hierarchy_dict(args.basic_hierarchy_file)
    else:
        hierarchy_dict = None
    durations_dict = duration2dict(args.durations)
    textgrid_fileslist =  glob.glob(args.input_dir + '/*.TextGrid', recursive=True)
    parent_textgrid_dict_intervals = {}
    parent_textgrid_dict_times = {}
    for textgrid_name in textgrid_fileslist:
        textgrid = TextGrid.fromFile(textgrid_name)
        basename = os.path.basename(textgrid_name).replace(".TextGrid", "")
        parent_name = basename.split("_")[0]
        parent_list = parent_textgrid_dict_intervals.get(parent_name, [])
        parent_time = parent_textgrid_dict_times.get(parent_name, 0)
        textgrid_interval = textgrid.getFirst(args.pred_tier)
        start_in_file = durations_dict[basename + ".wav"][0]
        for interval in textgrid_interval.intervals:
            if interval.mark:
                parent_list.append([round(interval.minTime + start_in_file, 3), round(interval.maxTime + start_in_file,3), interval.mark])
        parent_textgrid_dict_intervals[parent_name] = parent_list
        parent_textgrid_dict_times[parent_name] = max(textgrid.maxTime+start_in_file, parent_time)

    for base_filename, all_preds_intervals in parent_textgrid_dict_intervals.items():

        if hierarchy_dict:
            prev_hierarchy, prev_filename = get_hierarchy_path(hierarchy_dict, base_filename)
            new_dir = os.path.join(args.output_dir, prev_hierarchy)
            if not os.path.exists(new_dir):
                os.makedirs(new_dir)
            new_textgrid_path = os.path.join(new_dir, prev_filename.replace(".wav",".TextGrid")) 
        else:
            new_textgrid_path = os.path.join(args.output_dir, base_filename + ".TextGrid")

        sorted_all_preds_intervals = sorted(all_preds_intervals, key=lambda item: item[0])
        if args.use_prev_textgrid:
            hierarchy_filename = hierarchy_dict[base_filename]
            if ".wav" in hierarchy_filename:
                prev_textgrid = hierarchy_dict[base_filename].replace(".wav",".TextGrid")
            else:
                prev_textgrid = hierarchy_dict[base_filename].replace(".WAV",".TextGrid")
            textgrid = TextGrid.fromFile(prev_textgrid)
        else:
            textgrid = TextGrid()
        new_tier_name = args.pred_tier
        tmp_x_min = 0
        tmp_x_max = parent_textgrid_dict_times[base_filename]
        tier = IntervalTier(new_tier_name)
        for item in sorted_all_preds_intervals:
            x_min = item[0]
            x_max = item[1]
            mark = item[2]
            tier.add(x_min, x_max, mark)
            tmp_x_min = x_max
        if tmp_x_min < tmp_x_max:
            tier.add(tmp_x_min, tmp_x_max, "")

        textgrid.append(tier)
        textgrid.write(new_textgrid_path)
except Exception as e:
    print(f"failed to merging, Error:{e}")
    exit(1)