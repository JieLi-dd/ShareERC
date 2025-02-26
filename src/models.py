import torch
import torch.nn as nn
from modules.Transformer import  MELDTransEncoder, AdditiveAttention
from modules.CrossmodalTransformer import CrossModalTransformerEncoder
from transformers import RobertaModel
from transformers import BertModel
from modules.EmoAudio import EmoAudio
import torch.nn.functional as F
from modules.CrossModalTransformer_my import TransformerEncoderLayer

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

'''multimodal model'''
class MultiModalTransformerForClassification(nn.Module):

    def __init__(self, config):
        super(MultiModalTransformerForClassification,self).__init__()
        self.choice_modality = config.choice_modality
        self.num_labels = config.num_labels
        self.get_text_utt_max_lens = config.get_text_utt_max_lens
        self.hidden_size = config.hidden_size
        if config.pretrainedtextmodel_path.split('/')[-1] == 'roberta-large':
            self.text_pretrained_model = 'roberta'
        else:
            self.text_pretrained_model = 'bert'

        self.audio_emb_dim = config.audio_featExtr_dim
        self.audio_utt_Transformernum = config.audio_utt_Transformernum
        self.get_audio_utt_max_lens = config.get_audio_utt_max_lens
        
        #crossmodal transformer
        self.crossmodal_num_heads_TA = config.crossmodal_num_heads_TA
        self.crossmodal_layers_TA = config.crossmodal_layers_TA
        self.crossmodal_attn_dropout_TA = config.crossmodal_attn_dropout_TA
        
        '''Textual modality through RoBERTa or BERT'''
        if self.text_pretrained_model == 'roberta':
            self.roberta = RobertaModel.from_pretrained(config.pretrainedtextmodel_path)
            self.text_linear = nn.Linear(self.roberta.config.hidden_size, self.hidden_size)
        else:
            self.bert = BertModel.from_pretrained(config.pretrainedtextmodel_path)
            self.text_linear = nn.Linear(self.bert.config.hidden_size, self.hidden_size)


        self.audio_linear = nn.Linear(self.audio_emb_dim, self.hidden_size)
        self.audio_utt_transformer = MELDTransEncoder(config, self.audio_utt_Transformernum, self.get_audio_utt_max_lens, self.hidden_size)  #执行self-attention transformer


        self.attention = AdditiveAttention(self.hidden_size, self.hidden_size)

        self.CrossModalTrans_TA = CrossModalTransformerEncoder(self.hidden_size, self.crossmodal_num_heads_TA, self.crossmodal_layers_TA, self.crossmodal_attn_dropout_TA)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.classifier = nn.Linear(self.hidden_size, self.num_labels)

        # # 我添加的代码
        self.ffc = nn.Linear(self.hidden_size, self.hidden_size)
        self.text_attention = AdditiveAttention(self.hidden_size, self.hidden_size)
        self.audio_attention = AdditiveAttention(self.hidden_size, self.hidden_size)
        self.layernorm = nn.LayerNorm(self.hidden_size)
        self.classifier = nn.Linear(self.hidden_size, self.num_labels)
        self.emoaudio = EmoAudio.make_model(N=2, feat_dim=768, ff_dim=768*4, head=12, dropout=0.1)

    def forward(self, batch_text_input_ids=None, batch_text_input_mask=None, batch_text_sep_mask=None, 
                        audio_inputs=None, audio_mask=None,  batchUtt_in_dia_idx=None):

        if self.text_pretrained_model == 'roberta':
            #<s>utt_1</s></s>utt_2</s></s>utt_3</s>...
            outputs = self.roberta(batch_text_input_ids, batch_text_input_mask)
        else:
            #[CLS]utt_1[SEP]utt_2[SEP]utt_3[SEP]...
            outputs = self.bert(batch_text_input_ids, batch_text_input_mask)

        text_pretrained_model_out = outputs[0]  # (num_dia, max_sequence_len, 1024)
        text_utt_linear = self.text_linear(text_pretrained_model_out) #(num_dia, max_sequence_len, hidden_size)

        '''
        Extract word-level textual representations for each utterance.
        '''
        utt_batch_size = audio_inputs.shape[0]

        batch_text_feat_update = torch.zeros((utt_batch_size, self.get_text_utt_max_lens, text_utt_linear.shape[-1])).cuda()  #batch_size, max_utt_len, hidden_size
        batch_text_sep_mask_update = torch.zeros((utt_batch_size, self.get_text_utt_max_lens)).cuda() #batch_size, max_utt_len

        for i in range(utt_batch_size):
            curr_utt_in_dia_idx = batchUtt_in_dia_idx[i] #the position of the target utterance in the current dialogue.
            curr_dia_mask = batch_text_sep_mask[i]
            each_utt_index = []

            for index, value in enumerate(list(curr_dia_mask)):
                if value == 1:
                    each_utt_index.append(index)  #record the index position of the first </s> or [SEP] token for each utterance.
                    if curr_utt_in_dia_idx  == 0:
                        '''the current utterance is at the 0th position in the dialogue.'''
                        curr_utt_len = index-1   #remove the starting <s> and ending </s> tokens, or remove the starting [CLS] and ending [SEP] tokens.
                        if curr_utt_len > self.get_text_utt_max_lens:
                            curr_utt_len = self.get_text_utt_max_lens
                        batch_text_feat_update[i][:curr_utt_len] = text_utt_linear[i][1:curr_utt_len+1] #从<s>或者[CLS]之后开始
                        batch_text_sep_mask_update[i][:curr_utt_len] = 1
                        break
                    elif curr_utt_in_dia_idx >0 and curr_utt_in_dia_idx + 1 == len(each_utt_index):
                        curr_ut_id = len(each_utt_index) -1 #1
                        pre_ut_id = len(each_utt_index) - 2 #0
                        curr_ut = each_utt_index[curr_ut_id]
                        pre_ut = each_utt_index[pre_ut_id]
                        if self.text_pretrained_model == 'roberta':
                            curr_utt_len = curr_ut - pre_ut - 2  #remove </s> and <s>
                        else:
                            curr_utt_len = curr_ut - pre_ut - 1 #remove [SEP]

                        if curr_utt_len > self.get_text_utt_max_lens:
                            curr_utt_len = self.get_text_utt_max_lens
                        if self.text_pretrained_model == 'roberta':
                            batch_text_feat_update[i][:curr_utt_len] = text_utt_linear[i][pre_ut+2:pre_ut+2+curr_utt_len]  #从</s><s>之后开始
                        else:
                            batch_text_feat_update[i][:curr_utt_len] = text_utt_linear[i][pre_ut+1:pre_ut+1+curr_utt_len]  #从[SEP]之后开始
                        batch_text_sep_mask_update[i][:curr_utt_len] = 1
                        break
        #for memory
        del text_utt_linear, batch_text_sep_mask

        '''ACA'''
        #Input dim: (batch_size, Max_utt_len, pretrained_wav2vec_dim)
        audio_extended_utt_mask = audio_mask.unsqueeze(1).unsqueeze(2)
        audio_extended_utt_mask = (1.0 - audio_extended_utt_mask) * -10000.0
        audio_emb_linear = self.audio_linear(audio_inputs)
        audio_utt_trans = self.audio_utt_transformer(audio_emb_linear, audio_extended_utt_mask)    #(batch_size, utt_max_lens, self.hidden_size)

        # text_crossAudio_att = self.emoaudio(batch_text_feat_update, audio_inputs, batch_text_sep_mask_update, audio_mask)
        # audio_crossText_att = self.emoaudio(audio_inputs, batch_text_feat_update, audio_mask, batch_text_sep_mask_update)
        # text_crossAudio_att = self.layernorm(batch_text_feat_update)
        # audio_crossText_att = self.layernorm(audio_inputs)

        """CMS-Attention"""
        # # 我添加
        text_crossAudio_att = self.emoaudio(batch_text_feat_update, audio_utt_trans, batch_text_sep_mask_update, audio_mask)
        audio_crossText_att = self.emoaudio(audio_utt_trans, batch_text_feat_update, audio_mask, batch_text_sep_mask_update)

        text_crossAudio_att = self.layernorm(text_crossAudio_att)
        audio_crossText_att = self.layernorm(audio_crossText_att)

        textaudio_cross_feat = torch.concat((text_crossAudio_att.transpose(0, 1), audio_crossText_att.transpose(0, 1)), dim=0)
        final_utt_feat = textaudio_cross_feat.transpose(0,1)

        T_A_utt_mask = torch.concat((batch_text_sep_mask_update, audio_mask),dim=1)
        final_utt_mask = T_A_utt_mask

        # 极简模态融合机制
        final_utt_feat = self.ffc(final_utt_feat)
        final_utt_feat = self.layernorm(final_utt_feat)
        final_utt_feat = self.dropout(final_utt_feat)

        multimodal_out, _ = self.attention(final_utt_feat, final_utt_mask) #（batch_size, self.hidden_size）
        # multimodal_out = torch.max(final_utt_feat, dim=1)[0]

        # classify
        multimodal_output = self.dropout(multimodal_out)
        logits = self.classifier(multimodal_output)

        # text_out, _ = self.text_attention(batch_text_feat_update, batch_text_sep_mask_update)
        # text_output = self.dropout(text_out)
        # text_logits = self.classifier(text_output)
        # audio_out, _ = self.audio_attention(audio_utt_trans, audio_mask)
        # audio_output = self.dropout(audio_out)
        # audio_logits = self.classifier(audio_output)

        return logits
