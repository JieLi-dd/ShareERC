import torch
import os
from utils.dataset import loading_multimodal_data, loading_iemocap_data
import random
import numpy as np
import PIL

#---------------------------------------------------------------------------------------------------------------------------------------------#

def get_iemocap_data(args, split='train'):
    data = loading_iemocap_data(args, split)
    return data

def get_multimodal_data(args, split='train'):
    data_path = os.path.join(args.data_load_path, args.choice_modality, f'meld_multimodal_{split}_{args.choice_modality}_{args.plm_name}.dt')
    print('load MELD_multimodal_'+args.choice_modality+'_'+split+'...')
    if not os.path.exists(data_path):
        print(f"  - Creating new {split} data")
        data = loading_multimodal_data(args,split)
        # torch.save(data, data_path, pickle_protocol=4)
    else:
        print(f"  - Found cached {split} data")
        data = torch.load(data_path, map_location=torch.device('cpu'))
    return data

#---------------------------------------------------------------------------------------------------------------------------------------------#

load_project_path = os.path.abspath(os.path.dirname(__file__))


def save_Multimodal_model(model, args,curr_time):
    save_model_name = 'multimodal_model_{}_{}.pt'.format(args.choice_modality,curr_time)
    if not os.path.exists(args.save_Model_path):
        os.makedirs(args.save_Model_path)
    save_path = os.path.join(args.save_Model_path, save_model_name)
    torch.save(model, save_path, pickle_protocol=4)
    print(f"Saved model at saved_model/{save_model_name}!")

def load_Multimodal_model(choice_modality, save_Model_path,best_model_time):
    save_model_name = 'multimodal_model_{}_{}.pt'.format(choice_modality,best_model_time)
    load_path = os.path.join(save_Model_path, save_model_name)
    print('Loading the best Multimodal model for testing:'+save_model_name)
    model = torch.load(load_path)
    return model

#---------------------------------------------------------------------------------------------------------------------------------------------#
def save_Unimodal_model(model, args,curr_time):
    save_model_name = os.path.join(args.save_Model_path, 'unimodal_model_{}_{}.pt').format(args.choice_modality,curr_time)
    torch.save(model, save_model_name, pickle_protocol=4)
    print(f"Saved model at saved_model/unimodal_model_{args.choice_modality}_{curr_time}.pt!")

def load_Unimodal_model(choice_modality, save_Model_path,best_model_time):
    save_model_name = 'unimodal_model_{}_{}.pt'.format(choice_modality,best_model_time)
    load_path = os.path.join(save_Model_path, save_model_name)
    print('Loading the best unimodal model for testing:'+save_model_name)
    model = torch.load(load_path)
    return model

