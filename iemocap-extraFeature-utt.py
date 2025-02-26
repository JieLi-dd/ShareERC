# coding:utf-8

from cmath import e
# import cv2
from PIL import Image
import os
import os.path as osp
import sys
import pandas as pd
import numpy as np
from glob import glob
from tqdm import tqdm

import torch
import json
import shutil
import pickle
import math

sys.path.append(os.path.join(os.getcwd(), "../.."))

from torchvision import transforms

from audio import Wav2VecExtractor

# from make_ref import make_text


os.environ["CUDA_VISIBLE_DEVICES"] = '0'

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def make_text(csv_path, set_name):
    df = pd.read_csv(csv_path, encoding='utf8')
    ret = {}
    labels = []
    dia_utt_list = []
    audio_paths = []
    label_index = {'neu': 0, 'fru': 1, 'ang': 2, 'sad': 3, 'hap': 4, 'exc': 5}
    for _, row in df.iterrows():
        dia_num = int(row['Dialogue_ID'])
        utt_num = int(row['Utterance_ID'])
        utt_id = f'{set_name}/dia{dia_num}_utt{utt_num}'

        dia_utt_list.append(f'dia{dia_num}_utt{utt_num}')

        text = row['Utterance']

        ret[utt_id] = text
        label = row['Emotion']
        labels.append(label_index[label])

        audio_path = row['Wav_Path']
        audio_paths.append(audio_path)

    return ret, dia_utt_list, labels, audio_paths


class IEMOCAP():
    def __init__(self, modality, data_dir, pre_output_dir):

        self.modality = modality
        # data path
        self.data_dir = data_dir  #
        self.pre_output_dir = pre_output_dir  #

        # text path
        self.train_path = os.path.join(data_dir, 'IEMOCAP_train.csv')
        self.val_path = os.path.join(data_dir, 'IEMOCAP_dev.csv')
        self.test_path = os.path.join(data_dir, 'IEMOCAP_test.csv')

    def __padding(self, feature, MAX_LEN):

        length = feature.shape[0]
        if length >= MAX_LEN:
            return feature[:MAX_LEN, :], np.ones((MAX_LEN))

        pad = np.zeros([MAX_LEN - length, feature.shape[-1]])

        utt_pad = np.ones((length), dtype=int)

        utt_mask = np.zeros((MAX_LEN - length), dtype=int)

        feature = np.concatenate((feature, pad), axis=0)
        utt_fina_mask = np.concatenate((utt_pad, utt_mask), axis=0)

        return feature, utt_fina_mask

    def __paddingSequence(self, sequences):
        feature_dim = sequences[0].shape[-1]
        lens = [s.shape[0] for s in sequences]

        final_length = int(np.mean(lens) + 3 * np.std(lens))
        # padding sequences to final_length
        final_sequence = np.zeros([len(sequences), final_length, feature_dim])
        final_utt_mask = np.zeros([len(sequences), final_length])
        for i, s in enumerate(sequences):
            final_sequence[i], final_utt_mask[i] = self.__padding(s, final_length)
        return final_sequence, final_utt_mask

    def preprocess_data(self, modality=None, pretrainedAudioPath=None, AudioModelChoice=None, face_or_frame=None,
                        pretrained_vision_dataset=None, add_prob_distribution=None):
        self.data = {}
        self.data['train'] = {}
        self.data['test'] = {}
        self.data['dev'] = {}

        if modality == 'audio':
            print('开始执行语音模态的初始化特征提取！')
            pretrainedAudioPath = os.path.join(pretrainedAudioPath, AudioModelChoice)
            if AudioModelChoice == 'data2vec':
                extract_audio_feat = None  # Data2VecExtractor(pretrainedAudioPath,0)
            elif AudioModelChoice == 'wav2vec2.0':
                extract_audio_feat = Wav2VecExtractor(pretrainedAudioPath, 0)

        for set_name in ['dev', 'test']:
            csv_path = osp.join(self.data_dir, 'IEMOCAP_' + set_name + '.csv')
            _, int2name, labels, audio_paths = make_text(csv_path, set_name)
            features_utt = []
            label_Emotion = []

            i = 0
            j = 0

            for utt_id in tqdm(int2name):

                if utt_id == 'dia220_utt0':
                    continue

                profile = []
                emotion = np.array(labels[i])
                if modality == 'audio':
                    audio_path = audio_paths[i]
                    embedding = extract_audio_feat(audio_path)
                    features_utt.append(embedding)
                label_Emotion.append(emotion)
                i += 1

            feature_fina, utt_mask = self.__paddingSequence(
                features_utt)  # (num_utterance, max_utt_lens, featureExtractor_dim)

            label_Emotion = np.array(label_Emotion)

            if modality == 'audio':
                self.data[set_name]['audio'] = feature_fina
                self.data[set_name]['audio_utt_mask'] = utt_mask

            if modality == 'audio':
                save_path = 'preprocess_data/IEMOCAP_full_release/T+A/iemocap_{}_audio_utt.pkl'.format(set_name)

            f = open(save_path, 'wb')

            pickle.dump(self.data, f, protocol=4)
            f.close()


if __name__ == '__main__':

    data_dir = 'preprocess_data/IEMOCAP_full_release'

    pre_output_dir = 'preprocess_data/IEMOCAP_full_release'
    #############################################################################################

    modality = 'audio'  # audio, vision

    # add_prob_distribution = False  #是否为每帧图片给予一个初始化的重要性
    #############################################################################################

    iemocap = IEMOCAP(modality, data_dir, pre_output_dir)

    if modality == 'audio':
        '''语音模态预训练模型'''
        pretrainedAudioPath = 'pretrained_model/audio/'
        AudioModelChoice = 'wav2vec2.0'
        iemocap.preprocess_data(modality, pretrainedAudioPath, AudioModelChoice, None, None, False)


