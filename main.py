# -*- encoding: utf-8 -*-

import os
import torch
import argparse
from utils.util import *
from torch.utils.data import DataLoader,RandomSampler, SequentialSampler
from train import Lite
from torch.utils.data import TensorDataset


load_project_path = os.path.abspath(os.path.dirname(__file__))

parser = argparse.ArgumentParser(description='A Facial Expression-Aware Multimodal Multi-task Learning Framework for Emotion Recognition in Multi-party Conversations')

#---------------------------------------------------------------------------------------------------------------------------------------------#
'''MELD dataset loading'''
parser.add_argument('--load_anno_csv_path', type=str, default='/home/lijie/MasterStudy/ResDialogue/preprocess_data')
parser.add_argument('--meld_text_path', type=str, default='/home/lijie/MasterStudy/ResDialogue/preprocess_data')
parser.add_argument('--num_labels', type=int, default=7, help='classes number of meld')
parser.add_argument('--data_load_path', type=str, default=os.path.join(load_project_path,'preprocess_data/'),    
                    help='path for storing the data')
parser.add_argument('--save_Model_path', default=os.path.join(load_project_path,'saved_model')) 
parser.add_argument('--plm_name', type=str, default='roberta-large', choices='[roberta-large, bert-large]')
parser.add_argument('--choice_modality', type=str, default='T+A', choices='[T+A, T, A]')

#---------------------------------------------------------------------------------------------------------------------------------------------#
#tuning
parser.add_argument('--num_epochs', type=int, default=10, help='number of epochs')
parser.add_argument('--trg_lr', type=float, default=7e-6, help='initial learning rate')
parser.add_argument('--weight_decay', type=float, default=0.01, help='weight decay')
parser.add_argument('--warm_up', type=float, default=0.1, help='dynamic adjust learning rate')
parser.add_argument('--trg_batch_size', type=int, default=1, help='num of dialogues in MELD')
parser.add_argument('--trg_accumulation_steps',type=int, default=4, help='gradient accumulation for trg task')
#-------------------------------------------

#multi-modal fusion
parser.add_argument('--crossmodal_layers_TA', type=int, default=2, help='crossmodal layers of text and audio') 
parser.add_argument('--crossmodal_num_heads_TA', type=int, default=12)
parser.add_argument('--crossmodal_attn_dropout_TA', type=float, default=0.1, help='dropout applied on the attention weights')

#---------------------------------------------------------------------------------------------------------------------------------------------#
#self-attention transformer for audio and vision
parser.add_argument('--audio_utt_Transformernum',type=int, default=5, help='num of self-attention for audio')
parser.add_argument('--hidden_size', type=int, default=768, help='embedding size in the transformer, 768')
parser.add_argument('--num_attention_heads', type=int, default=12, help='number of heads for the transformer network, 12')  
parser.add_argument('--intermediate_size', type=int, default=3072, help='embedding intermediate layer size, 4*hidden_size, 3072')
parser.add_argument('--hidden_act', type=str, default='gelu', help='non-linear activation function')
parser.add_argument('--hidden_dropout_prob',type=float, default=0.1, help='multimodal dropout')
parser.add_argument('--attention_probs_dropout_prob',type=float, default=0.1,help='attention dropout')
parser.add_argument('--layer_norm_eps', type=float, default=1e-12, help='1e-12')  
parser.add_argument('--initializer_range',type=int, default=0.02) 
#---------------------------------------------------------------------------------------------------------------------------------------------#

parser.add_argument('--clip', type=float, default=0.8,  
                    help='gradient clip value (default: 0.8)')
parser.add_argument('--trg_log_interval', type=int, default=1600,
                    help='frequency of result logging')  
parser.add_argument('--seed', type=int, default=1111, help='random seed')
parser.add_argument('--num_classes', type=int, default=6, help='classes number of meld')
parser.add_argument('--radio', type=int, default=1, help='radio of the selected training data')
parser.add_argument('--temperature', type=float, default=0.08, help='temperature of the loss function')
parser.add_argument('--pool_size', type=int, default=512, help='size of the pool')
parser.add_argument('--support_set_size', type=int, default=64, help='size of the support set')
parser.add_argument('-dataset', type=str, default='MELD', choices='[MELD, IEMOCAP]', help='dataset name')

#---------------------------------------------------------------------------------------------------------------------------------------------#
#Evaluate the model on the test set directly
parser.add_argument('--doEval', type=bool, default=True, help='whether to evaluate the model on the test set directly')
# parser.add_argument('--load_unimodal_path', type=str, default='FacialMMT_unimodal/unimodal_model_V_06-16-01-14-14.pt',
#                     help='path to load the best unimodal to evaluate on the test set')
parser.add_argument('--load_unimodal_path', type=str, default='../saved_model/multimodal_model_T+A_07-01-00-28-42.pt',
                    help='path to load the best unimodal to evaluate on the test set') #../saved_model/multimodal_model_T+A_07-01-00-28-42.pt
parser.add_argument('--load_multimodal_path', type=str, default='../saved_model/multimodal_model_T+A_07-01-00-28-42.pt',
                    help='path to load the best multimodal to evaluate on the test set') # ../saved_model/multimodal_model_T+A_07-25-16-29-30.pt

args = parser.parse_args()
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    torch.set_default_tensor_type('torch.FloatTensor')


if args.choice_modality == 'T+A':
    if args.dataset == 'MELD':
        args.pretrainedtextmodel_path = os.path.join(load_project_path,'pretrained_model/public_pretrained_model',args.plm_name)
        trg_train_data = get_multimodal_data(args, 'train')
        trg_valid_data = get_multimodal_data(args, 'val')
        trg_test_data = get_multimodal_data(args, 'test')
    else:
        args.pretrainedtextmodel_path = os.path.join(load_project_path,'pretrained_model/public_pretrained_model',args.plm_name)
        trg_train_data = get_iemocap_data(args, 'train')
        trg_valid_data = get_iemocap_data(args, 'dev')
        trg_test_data = get_iemocap_data(args, 'test')

trg_train_loader = DataLoader(trg_train_data, sampler=RandomSampler(trg_train_data), batch_size=args.trg_batch_size)
trg_valid_loader = DataLoader(trg_valid_data, sampler=SequentialSampler(trg_valid_data), batch_size=args.trg_batch_size)
trg_test_loader = DataLoader(trg_test_data, sampler=SequentialSampler(trg_test_data), batch_size=args.trg_batch_size)

args.trg_n_train, args.trg_n_valid, args.trg_n_test = len(trg_train_data), len(trg_valid_data), len(trg_test_data)   


if args.choice_modality == 'T+A':
    args.audio_featExtr_dim = trg_train_data.get_audio_featExtr_dim()

if args.choice_modality == 'T+A':
    args.get_text_utt_max_lens = trg_train_data.get_text_max_utt_len()
    args.get_audio_utt_max_lens = max(trg_train_data.get_audio_max_utt_len(),trg_valid_data.get_audio_max_utt_len(),trg_test_data.get_audio_max_utt_len())

if __name__ == '__main__':
    print('&'*50)
    if args.doEval:
        print('Evaluating on the test set directly...')
        if args.choice_modality == 'T+A':
            Lite(strategy='dp', devices=1, accelerator="gpu", precision=16).run(args, None, None, trg_test_loader)
    else:
        print('Training from scratch...')
        if args.choice_modality == 'T+A':
            Lite(strategy='dp', devices=1, accelerator="gpu", precision=16).run(args,
                                                                                trg_train_loader,
                                                                                trg_valid_loader,
                                                                                trg_test_loader)


