import pandas as pd
from sentence_transformers import InputExample
from tqdm.auto import tqdm
import random
import torch
import numpy as np


class TripletDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, device='cpu', batch_size=10, shuffle=True, max_len=200):
        '''
        data: pandas dataframe with columns: ['triplet', 'positive_group', 'negative_group']
        tokenizer: tokenizer object from transformers library
        device: torch device
        batch_size: batch size for the dataloader
        shuffle: shuffle the data before batching
        '''
        self.data = data
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.device = device
        self.tokenizer = tokenizer
        self.max_len = max_len

        if shuffle:
            self.data = self.data.sample(frac=1).reset_index(drop=True)

        self.batched_data = []
        current_batch = []
        for k in tqdm(range(len(self.data)), unit='row', desc='Batching data'):
            row = self.data.iloc[k]
            item = row['triplet']
            current_batch.append(item.texts)
            if len(current_batch) == self.batch_size:
                self.batched_data.append(current_batch)
                current_batch = []
        for k in range(self.batch_size - len(current_batch) ):
            row = self.data.iloc[k]
            item = row['triplet']
            current_batch.append(item.texts)
        self.batched_data.append(current_batch)
        
        self.batched_data = np.array(self.batched_data)
        self.batched_anchors = []
        self.batched_positives = []
        self.batched_negatives = []
        
        for i in tqdm(range(len(self.batched_data)), unit='batch', desc='Tokenizing data'):
            batch = self.batched_data[i]
            anchors = batch[:, 0].tolist()
            positives = batch[:, 1].tolist()
            negatives = batch[:, 2].tolist()
            encoded_anchors   = self.tokenizer(anchors  , padding='max_length', max_length=self.max_len, truncation=True, return_tensors='pt')
            encoded_positives = self.tokenizer(positives, padding='max_length', max_length=self.max_len, truncation=True, return_tensors='pt')
            encoded_negatives = self.tokenizer(negatives, padding='max_length', max_length=self.max_len, truncation=True, return_tensors='pt')
            self.batched_anchors.append(encoded_anchors)
            self.batched_positives.append(encoded_positives)
            self.batched_negatives.append(encoded_negatives)

    def __len__(self):
        return len(self.batched_data)
    
    def __getitem__(self, idx):
        encoded_anchors   = self.batched_anchors[idx]
        encoded_positives = self.batched_positives[idx]
        encoded_negatives = self.batched_negatives[idx]
        encoded_anchors.to(self.device)
        encoded_positives.to(self.device)
        encoded_negatives.to(self.device)
        return encoded_anchors, encoded_positives, encoded_negatives

def get_sentence_id_label_df(path='./dataset/data.csv'):
    df = pd.read_csv(path)
    ids_to_labels_dict = get_ids_to_labels_dict(df)
    df = clean_data(df)
    #Loop over all sentences
    sentences = df['question'].tolist()
    ids = df['id'].tolist()
    labels = [ids_to_labels_dict[id] for id in ids]
    data = {'sentence': sentences, 'id': ids, 'label': labels}
    df = pd.DataFrame(data)
    return df

def get_ids_to_labels_dict(data):
    ids_to_labels_dict = {}
    grouped = data.groupby('id')
    for name, group in grouped:
        labels = group['node_name'].tolist()
        for label in labels:
            if isinstance(label, str):
                ids_to_labels_dict[name] = label
            if name in ids_to_labels_dict:
                break
        if name not in ids_to_labels_dict:
            ids_to_labels_dict[name] = f'unknown_{name}'
    return ids_to_labels_dict

def get_dataset(path='./dataset/data.csv', downsample_flag=True):
    df = pd.read_csv(path)
    df = clean_data(df)
    df = create_triplets(df)
    df = remove_duplicates(df)
    if downsample_flag:
        df = downsample(df)
    return df

def create_triplets(data):
    detailed_dict ={'triplet': [], 'positive_group': [], 'negative_group': [] }
    grouped = data.groupby('id')

    for name, group in tqdm(grouped, unit='group', desc='Creating triplets'):
        questions = group['question'].tolist()
        for i in range(len(questions)-1):
            anchor = questions[i]

            for j in range(len(questions)-1):
                if j == i:
                    continue
                positive = questions[j]

                for other_name, other_group in grouped:
                    if name == other_name:
                        continue
                    negatives = other_group['question'].tolist()

                    for negative in negatives:
                        triplet = InputExample(texts=[anchor, positive, negative])
                        detailed_dict['triplet'].append(triplet)
                        detailed_dict['positive_group'].append(name)
                        detailed_dict['negative_group'].append(other_name)

    data = pd.DataFrame(detailed_dict)
    return data

def downsample(data):
    min_len = data.positive_group.value_counts().min()
    grouped = data.groupby('positive_group')
    data = pd.concat([group.sample(min_len) for name, group in grouped])
    data = data.reset_index(drop=True)
    return data

def clean_data(data):
    data = data.dropna(subset=['question'])
    data = data.dropna(subset=['id'])
    data = data.reset_index(drop=True)
    return data

def remove_duplicates(data):
    data = data.drop_duplicates(subset=['triplet'])
    data = data.reset_index(drop=True)
    return data



