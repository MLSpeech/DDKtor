

__author__ = 'YaelSegal'

import torch
import torch.nn.functional as F
import argparse
import dataset 
import numpy as np
import os
from model import load_model
import glob
import random
from utils import SR, create_textgrid
import tqdm
import dataset
from helpers.utilities import basic_hierarchy_dict

parser = argparse.ArgumentParser(description='test vowel/vot')
parser.add_argument('--data', type=str, default='./data/processed/' , help="directory of the data",)
parser.add_argument('--out_dir', type=str, default='./data/out_tg/tmp_parts' , help="output directory",)

# trained on kasia data
parser.add_argument('--model', type=str, default='./model_cnn_lstm/data_KASIA_ntype_lstm_sim_lr_0.0001_input_size_256_num_layers_2_hidden_size_256_channels_256_normalize_True_norm_type_z_biLSTM_True_measure_rval_dropout_0.3_class_num_3_sigmoid_False_chain_bandreject,noise_lamda_1.0_59021734.pth', help='directory to save the model')

parser.add_argument('--basic_hierarchy_file', default="./data/raw/all_files/files.txt", type=str, help="features dir")
parser.add_argument('--cuda', action='store_true', help='use CUDA')
parser.add_argument('--seed', type=int, default=1245,	help='random seed')


args = parser.parse_args()

if not os.path.exists(args.out_dir):
    os.makedirs(args.out_dir)

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


args.cuda = args.cuda and torch.cuda.is_available()
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    device = 'cuda'
else:
    device = 'cpu'



########################################## testing ###########################################


path_model = args.model
test_model, normalize, sigmoid, norm_type  = load_model(path_model) 
if args.cuda:
    test_model = test_model.cuda()
hierarchy_dict = basic_hierarchy_dict(args.basic_hierarchy_file)
with torch.no_grad():
    test_model.eval()
    files_list = glob.glob(args.data + "/*.wav")

    for wav_filename in tqdm.tqdm(files_list):
        try:
            wav_basename = wav_filename.split("/")[-1]

            pred_dataset = dataset.PredictDataset(wav_filename,args.seed, slices_size=5000, overlap=0, normalize=normalize, norm_type=norm_type)
            pred_loader = torch.utils.data.DataLoader( pred_dataset, batch_size=100, shuffle=False,
            num_workers=0, pin_memory=args.cuda, collate_fn= dataset.PadCollatePred(dim=0))
            all_pred_class_idx = []
            conf = []
            for batch_idx, (raw, lens_list) in enumerate(pred_loader):
                raw = raw.to(device)

                hidden = test_model.init_hidden(raw.size(0), device)
                all_outputs = test_model(raw, hidden,lens_list)
                if len(all_outputs) == 2:
                    output , hidden = all_outputs
                else:
                    output, vector_out, hidden = all_outputs

                for idx in range(output.size(0)): 
                    cur_len = lens_list[idx]
                    pred_class_values, pred_class_idx = torch.max(output[idx,:cur_len], dim=1)
                    conf.extend(F.softmax(output[idx,:cur_len], dim=1).cpu().numpy().tolist())
                    pred_class_idx = pred_class_idx.cpu().numpy()
                    pred_class_values = pred_class_values.cpu().numpy()
                    all_pred_class_idx.extend(pred_class_idx)
        except Exception as e:
            base_filename = os.path.basename(wav_filename)
            hierarchy_num_array = base_filename.split("_")
            hierarchy_num_filename = hierarchy_num_array[0]
            window_number = hierarchy_num_array[1].replace(".wav", "")
            new_message = "fail to run model prediction on file: {}, window number: {}".format(hierarchy_dict[hierarchy_num_filename], window_number)
            raise Exception(new_message)

        textgrid_basename = wav_basename.replace(".wav", ".TextGrid")
        textgrid_filename = os.path.join(args.out_dir, textgrid_basename)
        create_textgrid(np.array(all_pred_class_idx), textgrid_filename, pred_dataset.wav_duration)








