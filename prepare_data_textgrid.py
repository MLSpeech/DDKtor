__author__ = 'YaelSegal'
import argparse
import os
import soundfile 
import glob
import tqdm
from helpers.textgrid import *
from utils import fix_extention

parser = argparse.ArgumentParser(description='copy all wav files from all sub dirs to out_dir')
parser.add_argument('--input_dir', type=str, help='Path of wavs dir',default="./data/raw/all_files")
parser.add_argument('--output_dir', type=str, help='Path to output dir',default="./data/processed")
parser.add_argument('--windows_tier',  type=str, default="window", help='windows tier in textgrid')

args = parser.parse_args()
VOICE_STARTS_FILENAME = "voice_starts.txt"
def process_textgrid_data(raw_dir, output_dir, window_tier):
    textgrid_list = glob.glob(raw_dir + "/*.TextGrid")
    all_WAV_files =  glob.glob(raw_dir + "/*.WAV")
    voice_start_file_path = os.path.join(raw_dir, VOICE_STARTS_FILENAME)
    files_start_list = open(voice_start_file_path, "w")
    for textgrid_file in tqdm.tqdm(textgrid_list):
        try:
            wav_file = textgrid_file.replace(".TextGrid", ".WAV")
            if wav_file in all_WAV_files:
                fix_extention(os.path.join(raw_dir, wav_file))  # convert "WAV" to "wav" if nessecary
                wav_file = wav_file.replace('WAV', 'wav')
            else:
                wav_file = textgrid_file.replace(".TextGrid", ".wav")
            y , sr = soundfile.read(wav_file)

            name = wav_file.split("/")[-1].replace(".wav","")
            textgrid = TextGrid.fromFile(textgrid_file)
            if not window_tier in textgrid.getNames():
                raise Exception("window tier {} doesn't exist in {}".format(window_tier, textgrid_file))
            tier_instances = textgrid.getFirst(window_tier)
            count = 0
            for item in tier_instances.intervals:
                if re.search(r'\S', item.mark):
                    new_wav_filename = output_dir + "/{}_{}.wav".format(name, count)
                    duration_str = "{}/{}_{}.wav,{},{}\n".format(output_dir, name, count, item.minTime,item.maxTime)
                    files_start_list.write(duration_str)
                    soundfile.write(new_wav_filename, y[int(item.minTime*sr):int(item.maxTime*sr)], sr)
                    count += 1   
   
        except Exception as e:
            print("Error in file {}".format(textgrid_file))
            raise e
        print("End process of {}".format(textgrid_file))
    files_start_list.close()


if not os.path.isdir(args.input_dir) or not os.path.isdir(args.output_dir):
    print("[Error] Wrong input or output Path ")
    exit()

print("Input path : '{}' \nOutput path : '{}'".format(args.input_dir, args.output_dir))

try:
    process_textgrid_data(args.input_dir, args.output_dir, args.windows_tier)

except Exception as e:
    print(f"failed to proccess file, Error:{e}")
    exit(1)