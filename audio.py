import os
import torch
import soundfile as sf  #之后还需要sudo apt install libsndfile1
import librosa
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2Model

import sys
sys.path.append(os.path.join(os.getcwd(), "../.."))

# from base_worker import BaseWorker



class Wav2VecExtractor(object):
    def __init__(self, pretrainedAudiopath, gpu):
        self.downsample = 4
        self.device = torch.device('cuda:{}'.format(gpu))
        print('[INFO] use asr based model')
        self.processor = Wav2Vec2Processor.from_pretrained(pretrainedAudiopath)
        self.model = Wav2Vec2Model.from_pretrained(pretrainedAudiopath).to(self.device)

    @staticmethod
    def read_audio(wav_path):
        try:
            speech, sr = librosa.load(wav_path, sr=16000)
        except Exception as e:
            print(f"Error loading audio: {str(e)}")
            speech = np.zeros(4000)  # 创建一个4000个元素的全零数组
            sr = 16000  # 返回16000Hz的采样率
        if speech.shape[0] > 300000:
            speech = speech[:300000]
        return speech, sr
        # return speech, sr

    def __call__(self, wav):
        input_values, sr = Wav2VecExtractor.read_audio(wav)
        input_values = self.processor(input_values, return_tensors="pt", sampling_rate=sr).input_values.to(self.device)
        with torch.no_grad():
            ft = self.model(input_values).last_hidden_state

        if self.downsample > 0:
            ft = torch.cat([
                torch.mean(ft[:, i:i+self.downsample], dim=1) for i in range(0, ft.shape[1], self.downsample)
            ], dim=0)
        return ft.cpu().numpy()


