import json
from torch.utils.data.dataset import Dataset
import pickle
import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
import logging

from src.meld_bert_extraText import MELD, IEMOCAP

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')



NORMAL_MEAN = [0.5, 0.5, 0.5]
NORMAL_STD = [0.5, 0.5, 0.5]
SWIN_IMG_SIZE = 224

# MAX_DIA_LEN = 33  #(test集合中dia17的个数)

TEXT_MAX_UTT_LEN = 38  #MELD数据集中最长的utterance文本长度规定为38 其实最长有90


#-------------------------------------------------------------------------------------------------------------------------------------------------------

'''加载多模态数据'''
class loading_multimodal_dataset(Dataset):
    def __init__(self, text_inputs, args, split_type):
        super(loading_multimodal_dataset, self).__init__()

        #改动
        labels = text_inputs['labels']
        text_inputs = text_inputs['features']

        dataset_path = args.data_load_path

        self.choice_modality = args.choice_modality
        self.split_type = split_type
        '''文本模态(整个dialogue下所有utterance串起来)输入'''
        self.text_input_ids = torch.tensor([f.input_ids for f in text_inputs], dtype=torch.long).cpu().detach() #(dia_num, max_sequence_len)
        self.text_input_mask = torch.tensor([f.input_mask for f in text_inputs], dtype=torch.long).cpu().detach() #(dia_num, max_sequence_len)
        self.text_sep_mask = torch.tensor([f.sep_mask for f in text_inputs], dtype=torch.long).cpu().detach() #(dia_num, max_sequence_len)

        '''语音模态输入'''
        audio_path = os.path.join(dataset_path, self.choice_modality, 'meld_'+split_type + '_audio_utt.pkl')
        openfile_a = open(audio_path, 'rb')
        audio_data = pickle.load(openfile_a)

        self.audio_feature = audio_data[split_type]['audio']  #(utt_num, max_audio_utt_len, extraFeature_dim) 
        self.audio_utterance_mask = audio_data[split_type]['audio_utt_mask'] #(utt_num, max_audio_utt_len) 有效的置1, 无效的置0
        self.audio_max_utt_len, self.audio_feat_dim = self.audio_feature.shape[1], self.audio_feature.shape[-1]
        openfile_a.close()

        utt_profile_path = os.path.join(dataset_path, self.choice_modality, split_type + '_utt_profile.json')
        with open(utt_profile_path,'r') as rd:
            self.utt_profile = json.load(rd)


        '''三模态统一标签信息'''
        self.labels = torch.tensor(labels, dtype=torch.long).cpu().detach()   #(num_utt, )
    
    def __len__(self):
        return self.audio_feature.shape[0]

    def get_text_max_utt_len(self):
        return TEXT_MAX_UTT_LEN 

    def get_audio_max_utt_len(self):
        return self.audio_max_utt_len #平均值+3*标准差

    def get_audio_featExtr_dim(self):  #通过wav2vec2.0获得的768维
        return self.audio_feat_dim


    def __getitem__(self, index):    

        '''
        首先得知道这个index对应是哪个utterance, 之后再去找到该dialogue的index,
        建立一个字典, key为utt_index, value为对应的utterance的名称、从属的dialogue名称, 该dialogue的编号、该utterance在该dialogue下的位置. 比如: {1:['dia0_utt0', 'dia0', 0, 3]}
        '''
        curr_utt_profile = self.utt_profile[str(index)]
        curr_utt_name, currUtt_belg_dia_name, currUtt_belg_dia_idx, currUtt_belg_dia_len, currUtt_in_dia_idx = curr_utt_profile
        
        '''text'''
        curr_text_input_ids  = self.text_input_ids[currUtt_belg_dia_idx]
        curr_text_input_mask = self.text_input_mask[currUtt_belg_dia_idx]
        curr_text_sep_mask = self.text_sep_mask[currUtt_belg_dia_idx]

        '''audio'''
        audio_inputs = self.audio_feature[index] #加载当前utterance     .shape为(utt_max_lens, Feature_extra_dim) 
        audio_mask = self.audio_utterance_mask[index]  #(utt_max_lens)

        '''label'''
        curr_label_ids = self.labels[index]

        return curr_text_input_ids, curr_text_input_mask, curr_text_sep_mask, audio_inputs, audio_mask, curr_label_ids, currUtt_in_dia_idx


class loading_iemocap_dataset(Dataset):
    def __init__(self, text_inputs, args, split_type):
        super(loading_iemocap_dataset, self).__init__()

        # 改动
        labels = text_inputs['labels']
        text_inputs = text_inputs['features']

        dataset_path = os.path.join(args.data_load_path, 'IEMOCAP_full_release')

        self.choice_modality = args.choice_modality
        self.split_type = split_type
        '''文本模态(整个dialogue下所有utterance串起来)输入'''
        self.text_input_ids = torch.tensor([f.input_ids for f in text_inputs],
                                           dtype=torch.long).cpu().detach()  # (dia_num, max_sequence_len)
        self.text_input_mask = torch.tensor([f.input_mask for f in text_inputs],
                                            dtype=torch.long).cpu().detach()  # (dia_num, max_sequence_len)
        self.text_sep_mask = torch.tensor([f.sep_mask for f in text_inputs],
                                          dtype=torch.long).cpu().detach()  # (dia_num, max_sequence_len)

        '''语音模态输入'''
        audio_path = os.path.join(dataset_path, self.choice_modality, 'iemocap_' + split_type + '_audio_utt.pkl')
        openfile_a = open(audio_path, 'rb')
        audio_data = pickle.load(openfile_a)

        self.audio_feature = audio_data[split_type]['audio']  # (utt_num, max_audio_utt_len, extraFeature_dim)
        self.audio_utterance_mask = audio_data[split_type][
            'audio_utt_mask']  # (utt_num, max_audio_utt_len) 有效的置1, 无效的置0
        self.audio_max_utt_len, self.audio_feat_dim = self.audio_feature.shape[1], self.audio_feature.shape[-1]
        openfile_a.close()

        utt_profile_path = os.path.join(dataset_path, self.choice_modality, split_type + '_utt_profile.json')
        with open(utt_profile_path, 'r') as rd:
            self.utt_profile = json.load(rd)

        '''三模态统一标签信息'''
        self.labels = torch.tensor(labels, dtype=torch.long).cpu().detach()  # (num_utt, )

    def __len__(self):
        return self.audio_feature.shape[0]

    def get_text_max_utt_len(self):
        return TEXT_MAX_UTT_LEN

    def get_audio_max_utt_len(self):
        return self.audio_max_utt_len  # 平均值+3*标准差

    def get_audio_featExtr_dim(self):  # 通过wav2vec2.0获得的768维
        return self.audio_feat_dim

    def __getitem__(self, index):
        '''
        首先得知道这个index对应是哪个utterance, 之后再去找到该dialogue的index,
        建立一个字典, key为utt_index, value为对应的utterance的名称、从属的dialogue名称, 该dialogue的编号、该utterance在该dialogue下的位置. 比如: {1:['dia0_utt0', 'dia0', 0, 3]}
        '''
        curr_utt_profile = self.utt_profile[str(index)]
        curr_utt_name, currUtt_belg_dia_name, currUtt_belg_dia_idx, currUtt_belg_dia_len, currUtt_in_dia_idx = curr_utt_profile

        '''text'''
        curr_text_input_ids = self.text_input_ids[currUtt_belg_dia_idx]
        curr_text_input_mask = self.text_input_mask[currUtt_belg_dia_idx]
        curr_text_sep_mask = self.text_sep_mask[currUtt_belg_dia_idx]

        '''audio'''
        audio_inputs = self.audio_feature[index]  # 加载当前utterance     .shape为(utt_max_lens, Feature_extra_dim)
        audio_mask = self.audio_utterance_mask[index]  # (utt_max_lens)

        '''label'''
        curr_label_ids = self.labels[index]

        return curr_text_input_ids, curr_text_input_mask, curr_text_sep_mask, audio_inputs, audio_mask, curr_label_ids, currUtt_in_dia_idx

def loading_multimodal_data(args, split=None):

    load_anno_csv = args.load_anno_csv_path

    meld_text_path = args.meld_text_path

    meld = MELD(load_anno_csv, args.pretrainedtextmodel_path, meld_text_path, split)

    meld_text_features = meld.preprocess_data()  

    final_data = loading_multimodal_dataset(meld_text_features, args, split)

    return final_data


def loading_iemocap_data(args, split=None):
    load_anno_csv = args.load_anno_csv_path + '/IEMOCAP_full_release'
    iemocap_text_path = args.meld_text_path + '/IEMOCAP_full_release'
    iemocap = IEMOCAP(load_anno_csv, args.pretrainedtextmodel_path, iemocap_text_path, split)
    iemocap_text_features = iemocap.preprocess_data()
    final_data = loading_iemocap_dataset(iemocap_text_features, args, split)
    return final_data


def dist(x, y):

    return (1-F.cosine_similarity(x, y, dim=-1))/2 + 1e-8

def get_cluster_reps(model, data_loader, args):
    model.eval()
    results = []
    label_results = []

    with torch.no_grad():
        for i_batch, batch in tqdm(enumerate(data_loader), desc='generate representations for all data'):
            batch_size = args.trg_batch_size
            batch_text_input_ids, batch_text_input_mask, batch_text_sep_mask, audio_inputs, audio_mask, batch_label_ids, batchUtt_in_dia_idx = batch
            logits, outputs = model(batch_text_input_ids, batch_text_input_mask, batch_text_sep_mask, audio_inputs, audio_mask, batchUtt_in_dia_idx)

            for idx, label in enumerate(batch_label_ids):
                results.append(outputs[idx])
                label_results.append(label)

        def score_func(x, y):
            return (1 + F.cosine_similarity(x, y, dim=-1)) / 2 + 1e-8
    dim = results[0].shape[-1]
    results = torch.stack(results, 0).reshape(-1, dim)
    label_results = torch.stack(label_results, 0).reshape(-1)
    return results, label_results


def cluster(args, reps, labels, epoch=0):
    label_space = {}
    label_space_dataid = {}
    centers = []
    for idx in range(args.num_classes):
        label_space[idx] = []
        label_space_dataid[idx] = []
    for idx, turn_reps in enumerate(reps):
        emotion_label = labels[idx].item()
        if emotion_label < 0:
            continue

        label_space[emotion_label].append(turn_reps)
        label_space_dataid[emotion_label].append(idx)
    # clustering for each emotion class
    dim = label_space[0][0].shape[-1]

    max_num_clusters = 0
    cluster2dataid = {}
    cluster2classid = {}
    total_clusters = 0
    all_centers = []
    for emotion_label in range(args.num_classes):

        x = torch.stack(label_space[emotion_label], 0).reshape(-1, dim)
        num_clusters = 1
        cluster_idxs = torch.zeros(x.shape[0]).long()
        cluster_centers = x.mean(0).unsqueeze(0).cpu()
        logging.info('{} clusters for emotion {}'.format(num_clusters, emotion_label))
        centers.append(cluster_centers)

        max_num_clusters = max(num_clusters, max_num_clusters)
        # 记录聚类中心到数据索引的映射，由此来构造对比学习的样本
        cluster_idxs += total_clusters
        for d_idx, c_idx in enumerate(cluster_idxs.numpy().tolist()):  # d是索引，c是标签类别
            if c_idx < 0:
                continue
            if cluster2dataid.get(c_idx) is None:
                cluster2dataid[c_idx] = []
            cluster2classid[c_idx] = emotion_label
            cluster2dataid[c_idx].append(
                label_space_dataid[emotion_label][d_idx])
        total_clusters += num_clusters
        for c_idx in range(num_clusters):
            all_centers.append(cluster_centers[c_idx, :])

    centers_mask = []
    for emotion_label in range(args.num_classes):
        num_clusters, dim = centers[emotion_label].shape[0], centers[
            emotion_label].shape[-1]
        centers_mask.append(torch.zeros(max_num_clusters))
        centers_mask[emotion_label][:num_clusters] = 1
        centers[emotion_label] = torch.cat(
            (centers[emotion_label],
             torch.ones(max_num_clusters - num_clusters, dim)), 0)
    centers = torch.stack(centers, 0).to('cuda')
    centers_mask = torch.stack(centers_mask, 0).to('cuda')
    return centers, centers_mask, cluster2dataid, cluster2classid, all_centers

def gen_cl_data(args,
                reps,
                all_centers,
                cluster2dataid,
                cluster2classid,
                epoch=0):

    batch_size = args.trg_batch_size
    num_data = reps.shape[0]
    dim = reps.shape[-1]
    total_cluster = len(all_centers)

    cluster_idxs = torch.zeros(num_data).long()
    labels = torch.zeros(num_data).long()

    for c_idx in range(total_cluster):
        for data_id in cluster2dataid[c_idx]:
            cluster_idxs[data_id] = c_idx
            labels[data_id] = cluster2classid[c_idx]
    seed_list = selection(reps, all_centers, cluster2dataid, args.radio)

    # plot_data(reps, labels, epoch, seed_list)
    return seed_list, cluster_idxs


def selection(reps, all_centers, cluster2dataid, selection_ratio):
    total_cluster = len(all_centers)
    data2clusterid = {}
    for c_idx in range(total_cluster):
        for data_id in cluster2dataid[c_idx]:
            data2clusterid[data_id] = c_idx
    all_centers = torch.stack(all_centers, 0).to(reps.device)
    # difficult measure function
    dis_scores = []
    for idx, rep in enumerate(reps):
        self_center = all_centers[data2clusterid[idx]]
        self_dis = dist(rep, self_center)
        sum_dis = dist(
            rep.unsqueeze(0).expand_as(all_centers),
            all_centers
        )
        dis_scores.append(self_dis / sum_dis.sum())
    dis_scores = torch.FloatTensor(dis_scores)
    priority_seq = torch.argsort(dis_scores, descending=False).cpu().numpy().tolist()  # 困难度排序

    num_selection = int(selection_ratio * len(priority_seq))
    select_data_idx = priority_seq[:num_selection]

    return select_data_idx
