import pandas as pd
from sentence_transformers import InputExample
from tqdm.auto import tqdm
import random
import torch
import numpy as np
import gzip
import csv
import pickle
import os

class TripletDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, device='cpu', batch_size=10, shuffle=True, max_len=200, use_allnli=False):
        '''
        data: pandas dataframe with columns: ['triplet', 'positive_group', 'negative_group']
        tokenizer: tokenizer object from transformers library
        device: torch device
        batch_size: batch size for the dataloader
        shuffle: shuffle the data before batching
        '''
        self.using_allnli = use_allnli
        if self.using_allnli:
            # self.data = get_ALLNLI_dataset()
            big_allnli = get_ALLNLI_dataset()
            self.data = big_allnli[:5000]
            
        else:
            self.data = data
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.device = device
        self.tokenizer = tokenizer
        self.max_len = max_len

        if shuffle:
            if self.using_allnli:
                random.shuffle(self.data)
            else:
                self.data = self.data.sample(frac=1).reset_index(drop=True)

        self.batched_data = []
        current_batch = []
        for k in tqdm(range(len(self.data)), unit='row', desc='Batching data'):
            if self.using_allnli:
                item = self.data[k]
            else:
                row = self.data.iloc[k]
                item = row['triplet']
            current_batch.append(item.texts)
            if len(current_batch) == self.batch_size:
                self.batched_data.append(current_batch)
                current_batch = []
        for k in range(self.batch_size - len(current_batch) ):
            if self.using_allnli:
                item = self.data[k]
            else:
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



def get_ALLNLI_dataset():
    nli_dataset_path = 'dataset/AllNLI.tsv.gz'
    nli_dataset_pickle_path = 'dataset/AllNLI.pickle'
    if os.path.exists(nli_dataset_pickle_path):
        with open(nli_dataset_pickle_path, 'rb') as fIn:
            train_samples = pickle.load(fIn)
            return train_samples
    train_data = {}
    def add_to_samples(sent1, sent2, label):
        if sent1 not in train_data:
            train_data[sent1] = {'contradiction': set(), 'entailment': set(), 'neutral': set()}
        train_data[sent1][label].add(sent2)

    with gzip.open(nli_dataset_path, 'rt', encoding='utf8') as fIn:
        reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
        for row in reader:
            if row['split'] == 'train':
                sent1 = row['sentence1'].strip()
                sent2 = row['sentence2'].strip()

                add_to_samples(sent1, sent2, row['label'])
                add_to_samples(sent2, sent1, row['label'])  #Also add the opposite
    train_samples = []
    for sent1, others in train_data.items():
        if len(others['entailment']) > 0 and len(others['contradiction']) > 0:
            train_samples.append(InputExample(texts=[sent1, random.choice(list(others['entailment'])), random.choice(list(others['contradiction']))]))
            train_samples.append(InputExample(texts=[random.choice(list(others['entailment'])), sent1, random.choice(list(others['contradiction']))]))

    with open(nli_dataset_pickle_path, 'wb') as fOut:
        pickle.dump(train_samples, fOut)
    return train_samples
