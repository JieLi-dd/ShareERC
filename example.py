import pandas as pd
import json
import os

root_file_path = 'preprocess_data/IEMOCAP_full_release'
# for split in ['train', 'dev', 'test']:
#     file_path = os.path.join(root_file_path, f'IEMOCAP_{split}.csv')
#     df = pd.read_csv(file_path)
#     df['new_Dialogue_ID'] = ''
#     df['new_Utterance_ID'] = ''
#     pre_dia = 0
#     dia_num = 0
#     utt_num = 0
#     for i in range(len(df)):
#         dia = int(df.loc[i, 'Dialogue_ID'])
#         utt = int(df.loc[i, 'Utterance_ID'])
#         if dia == pre_dia:
#             if utt_num < 19:
#                 df.loc[i, 'new_Dialogue_ID'] = dia_num
#                 df.loc[i, 'new_Utterance_ID'] = utt_num
#                 utt_num += 1
#             else:
#                 dia_num += 1
#                 utt_num = 0
#                 df.loc[i, 'new_Dialogue_ID'] = dia_num
#                 df.loc[i, 'new_Utterance_ID'] = utt_num
#                 utt_num += 1
#         else:
#             dia_num += 1
#             utt_num = 0
#             df.loc[i, 'new_Dialogue_ID'] = dia_num
#             df.loc[i, 'new_Utterance_ID'] = utt_num
#             pre_dia = dia
#             utt_num += 1
#     output_file_path = os.path.join(root_file_path, f'IEMOCAP_{split}_updated.csv')
#     df.to_csv(output_file_path, index=False)
#     print(f'{split} done')
# print('all done')





# import pandas as pd
# import json
# import os
#
# root_file_path = 'preprocess_data/IEMOCAP_full_release'
# for split in ['train', 'dev', 'test']:
#     file_path = os.path.join(root_file_path, f'IEMOCAP_{split}.csv')
#     json_file_path = os.path.join(root_file_path, 'T+A', f'{split}_max_len.json')
#     df = pd.read_csv(file_path)
#     with open(json_file_path, 'r', encoding='utf-8') as file:
#         # 使用 json.load() 加载 JSON 数据
#         data = json.load(file)
#     profile = {}
#     for i in range(len(df)):
#         dia = int(df.loc[i, 'Dialogue_ID'])
#         utt = int(df.loc[i, 'Utterance_ID'])
#         dia_utt = f'dia{dia}_utt{utt}'
#         dia_num = f'dia{dia}'
#         max_len = data[f'{dia}']
#         profile[i] = [dia_utt, dia_num, dia, max_len, utt]
#     output_path = os.path.join(root_file_path, 'T+A', f'{split}_utt_profile.json')
#     with open(output_path, 'w', encoding='utf-8') as json_file:
#         json.dump(profile, json_file, ensure_ascii=False, indent=4)
#     print(f'{split} done')
# print('all done')







# # 计算最大长度
# for split in ['train', 'dev', 'test']:
#     file_path = os.path.join(root_file_path, f'IEMOCAP_{split}.csv')
#     df = pd.read_csv(file_path)
#
#     dia_max_len = {}
#     for i in range(len(df)):
#         dia = int(df.loc[i, 'Dialogue_ID'])
#         utt = int(df.loc[i, 'Utterance_ID'])
#         dia_max_len[dia] = max(utt, dia_max_len.get(dia, 0))
#     file_path = os.path.join(root_file_path, 'T+A', f'{split}_max_len.json')
#     with open(file_path, 'w', encoding='utf-8') as json_file:
#         json.dump(dia_max_len, json_file, ensure_ascii=False, indent=4)
#     print(f'{split} done')
#
# print('all done')






# # 设置train_text.json
# # 定义可能包含整数的列名
# int_columns = ['Dialogue_ID', 'Utterance_ID', 'label']
#
# for split in ['train', 'dev', 'test']:
#     file_path = os.path.join(root_file_path, f'IEMOCAP_{split}.csv')
#     df = pd.read_csv(file_path)
#
#     # 将指定列转换为 Python int 类型
#     df[int_columns] = df[int_columns].astype(int)
#
#     text = {}
#     for i in range(len(df)):
#         dia = int(df.loc[i, 'Dialogue_ID'])
#         utt = int(df.loc[i, 'Utterance_ID'])
#         label = int(df.loc[i, 'label'])
#         txt = df.loc[i, 'Utterance']
#         dia_utt = {'txt': [txt], 'label': label}
#         id = f'dia{dia}_utt{utt}'
#         text[id] = dia_utt
#
#     file_path = os.path.join(root_file_path, f'{split}_text.json')
#     with open(file_path, 'w', encoding='utf-8') as json_file:
#         json.dump(text, json_file, ensure_ascii=False, indent=4)
#     print(f'{split} done')
# print('all done')



# 设置lable2id
# for split in ['train', 'dev', 'test']:
#     file_path = os.path.join(root_file_path, f'IEMOCAP_{split}.csv')
#     df = pd.read_csv(file_path)
#
#     json_path = os.path.join(root_file_path, 'label2id.json')
#     with open(json_path, 'r', encoding='utf-8') as file:
#         data = json.load(file)
#     df['label'] = ''
#     for i in range(len(df)):
#         emo = df.loc[i, 'Emotion']
#         df.loc[i, 'label'] = data[emo]
#
#     df.to_csv(file_path, index=False)
#     print(f'{split} done')
# print('all done')


