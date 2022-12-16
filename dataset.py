import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pickle, pandas as pd
import json
import numpy as np
import random
from pandas import DataFrame
from transformers import AutoTokenizer

"""
    dataset.py: define your Dataset class here.
"""


class SST2(Dataset):
    def __init__(self, dataset_name='sst2', split="train"):
        dataset = self.read(dataset_name)[split]  # ('sentence', 'label', 'att_mask', 'input_ids', xxx)
        # (input_ids, attention_mask, token_type_ids, label, sentence)
        self.data = list(zip(dataset['input_ids'], dataset['attention_mask'], dataset['token_type_ids'],
                             dataset['label'], dataset['sentence']))

    def read(self, dataset_name):
        raw_dataset = load_dataset("glue", dataset_name)  # ('sentence', 'label', 'idx')
        ckpt = "bert-base-uncased"
        tokenizer = AutoTokenizer.from_pretrained(ckpt)

        def tokenize_fun(examples):
            return tokenizer(examples['sentence'], truncation=True)

        tokenized_dataset = raw_dataset.map(tokenize_fun, batched=True)  # add ('input_ids', 'att_mask', xxx)
        return tokenized_dataset

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

    def collate_fn(self, datas):
        """
        @params:
        datas: a list of data: (input_ids, att_mask, token_type_ids, label, sentence)
        """
        # input_ids, pad_token_id of bert-uncased is 0.
        input_ids = (pad_sequence([torch.LongTensor(d[0]) for d in datas], padding_value=0, batch_first=True)) # B x M
        # padding att_mask is 0
        att_mask = (pad_sequence([torch.LongTensor(d[1]) for d in datas], padding_value=0, batch_first=True)) # B x M
        token_type_ids = (pad_sequence([torch.LongTensor(d[2]) for d in datas], padding_value=1, batch_first=True)) # B x M
        label = torch.LongTensor([d[3] for d in datas]) # B
        sentence = [d[4] for d in datas] # B, a list of strings
        return input_ids, att_mask, token_type_ids, label, sentence




class IEMOCAPDataset(Dataset):

    def __init__(self, dataset_name='IEMOCAP', split='train', speaker_vocab=None, label_vocab=None, args=None,
                 tokenizer=None):
        self.speaker_vocab = speaker_vocab
        self.label_vocab = label_vocab
        self.args = args
        self.data = self.read(dataset_name, split, tokenizer)
        print(len(self.data))

        self.len = len(self.data)

    def read(self, dataset_name, split, tokenizer):
        with open('./data/%s/%s_data_roberta_v2.json.feature' % (dataset_name, split), encoding='utf-8') as f:
            raw_data = json.load(f)

        # process dialogue
        dialogs = []
        # raw_data = sorted(raw_data, key=lambda x:len(x))
        for d in raw_data:
            # if len(d) < 5 or len(d) > 6:
            #     continue
            utterances = []
            labels = []
            speakers = []
            features = []
            for i, u in enumerate(d):
                utterances.append(u['text'])
                labels.append(self.label_vocab['stoi'][u['label']] if 'label' in u.keys() else -1)
                speakers.append(self.speaker_vocab['stoi'][u['speaker']])
                features.append(u['cls'])
            dialogs.append({
                'utterances': utterances,
                'labels': labels,
                'speakers': speakers,
                'features': features
            })
        random.shuffle(dialogs)
        return dialogs

    def __getitem__(self, index):
        '''
        :param index:
        :return:
            feature,
            label
            speaker
            length
            text
        '''
        return torch.FloatTensor(self.data[index]['features']), \
               torch.LongTensor(self.data[index]['labels']), \
               self.data[index]['speakers'], \
               len(self.data[index]['labels']), \
               self.data[index]['utterances']

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        '''
        :param data:
            features, labels, speakers, length, utterances
        :return:
            features: (B, N, D) padded
            labels: (B, N) padded
            adj: (B, N, N) adj[:,i,:] means the direct predecessors of node i
            s_mask: (B, N, N) s_mask[:,i,:] means the speaker informations for predecessors of node i, where 1 denotes the same speaker, 0 denotes the different speaker
            lengths: (B, )
            utterances:  not a tensor
        '''
        max_dialog_len = max([d[3] for d in data])
        feaures = pad_sequence([d[0] for d in data], batch_first=True)  # (B, N, D)
        labels = pad_sequence([d[1] for d in data], batch_first=True, padding_value=-1)  # (B, N )
        # adj = self.get_adj_v1([d[2] for d in data], max_dialog_len)
        # s_mask, s_mask_onehot = self.get_s_mask([d[2] for d in data], max_dialog_len)
        lengths = torch.LongTensor([d[3] for d in data])
        speakers = pad_sequence([torch.LongTensor(d[2]) for d in data], batch_first=True, padding_value=-1)  # B x N
        utterances = [d[4] for d in data]

        return feaures, labels, speakers, lengths, utterances


if __name__ == '__main__':
    train_set = SST2()
    print(train_set[0])
