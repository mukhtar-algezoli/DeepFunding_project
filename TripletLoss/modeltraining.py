# import os
# work_dir = '/content/drive/MyDrive/DeepFund/'
# os.chdir(work_dir)
# os.chdir('DeepFunding_project/TripletLoss')

import os
import sys
import wandb
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm.auto import tqdm
from peft import LoraConfig
from network import get_sts_model
from triplet_dataset import get_dataset, get_sentence_id_label_df, TripletDataset
from model_evaluation import  calculate_dsiatances_from_embeddings, calculate_accuracy_from_embeddings


def triplets_df_to_single_df(triplets_df):
    triplets = triplets_df['triplet'].tolist()
    triplets = [triplet.texts for triplet in triplets]
    positive_groups = triplets_df['positive_group'].tolist()
    negative_groups = triplets_df['negative_group'].tolist()
    sentences =[]
    groups = []
    for triplet, positive_group, negative_group in zip(triplets, positive_groups, negative_groups):
        anchor, positive, negative = triplet
        if anchor not in sentences:
            sentences.append(anchor)
            groups.append(positive_group)
        if positive not in sentences:
            sentences.append(positive)
            groups.append(positive_group)
        if negative not in sentences:
            sentences.append(negative)
            groups.append(negative_group)
    data = {'sentence': sentences, 'group': groups}
    df = pd.DataFrame(data)
    return df

def main(args_dict, use_argparse=False):
    default_args_dict = {
        'model_path': 'ammarnasr/LoRa_all-MiniLM-L12-v1',
        'data_path': './dataset/data.csv',
        'device': 'cuda',
        'peft_config': None,
        'batch_size': 16,
        'lr': 1e-5,
        'triplet_loss': None,
        'num_epochs': 10,
        'max_len': 200,
        'eval_every': 150,
        'save_model_every': 500,
        'shuffle': True,
        'eval_data_path': None,
        'save_model_path': './models/LoRa',
        'model_save_name': None,
        'wandb_project_name': None,
    }
    for key in default_args_dict.keys():
        if key not in args_dict.keys():
            args_dict[key] = default_args_dict[key]
    #Pretty print args
    print('Arguments:')
    for key in args_dict.keys():
        print(f'{key}: {args_dict[key]}')
    train(**args_dict)

def get_train_eval_test_data(data_path, random_state=42):
    #Read CSV and drop nan values
    df = pd.read_csv(data_path)
    df = df.dropna(subset=['question'])
    df = df.dropna(subset=['id'])
    df = df.reset_index(drop=True)
    # Split data into train, val, and test 0.8, 0.1, 0.1
    train_df = df.sample(frac=0.8, random_state=random_state)
    test_df = df.drop(train_df.index)
    val_df = test_df.sample(frac=0.5, random_state=random_state)
    test_df = test_df.drop(val_df.index)
    # Reset indexes
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    #Get Triplets Datasets
    train_df = get_dataset(df=train_df)
    val_df = get_dataset(df=val_df)
    test_df = get_dataset(df=test_df)
    return train_df, val_df, test_df




def train(model_path, data_path='./dataset/data.csv', device='cuda', peft_config=None,
          batch_size=16, lr=1e-5, triplet_loss=None, num_epochs=5, max_len=100,
          eval_every=100,save_model_every=1000, shuffle=True, eval_data_path=None,
          save_model_path='./models/LoRa', model_save_name=None, wandb_project_name=None):
    
    print('Loading model...')
    model = get_sts_model(model_path, device, peft_config)
    tokenizer = model.tokenizer
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=lr)
    if not os.path.exists(save_model_path):
        os.makedirs(save_model_path)
    if model_save_name is None:
        model_save_name = model_path.split('/')[-1]
    if triplet_loss is None:
        triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
    if eval_data_path is None:
        eval_data_path = data_path
    if wandb_project_name is None:
        wandb_project_name = model_save_name+'-tracking'

    print('Loading data...')
    train_df, val_df, test_df = get_train_eval_test_data(data_path)
    val_single_df = triplets_df_to_single_df(val_df)
    eval_dataset = TripletDataset(val_df, tokenizer=tokenizer, device=device, batch_size=batch_size, shuffle=shuffle, max_len=max_len)

    print('Initializing wandb...')
    wandb.init(project=wandb_project_name)
    wandb.config.update(
        {
            'model_path': model_path,
            'data_path': data_path,
            'device': device,
            'LoRa_Rank': peft_config.r,
            'LoRa_Alpha': peft_config.lora_alpha,
            'LoRa_Dropout': peft_config.lora_dropout,
            'LoRa_Target_Modules': peft_config.target_modules,
            'batch_size': batch_size,
            'lr': lr,
            'triplet_loss': triplet_loss,
            'num_epochs': num_epochs,
            'max_len': max_len,
            'eval_every': eval_every,
            'save_model_every': save_model_every,
            'shuffle': shuffle,
            'eval_data_path': eval_data_path,
            'save_model_path': save_model_path,
            'model_save_name': model_save_name,
            'wandb_project_name': wandb_project_name
        }
    )
    wandb.watch(model)



    print('Training model...')
    epochs_tbar = tqdm(range(num_epochs), unit='epoch')
    steps = 0
    accuracy = 0
    for epoch in epochs_tbar:
        train_dataset = TripletDataset(train_df, tokenizer=tokenizer, device=device, batch_size=batch_size, shuffle=shuffle, max_len=max_len)
        epoch_steps = 0
        accumelated_loss = 0
        batches_tbar = tqdm(train_dataset, unit='batch')
        for input in batches_tbar:
                steps += 1
                epoch_steps += 1
                anchor = model(input[0])
                positive = model(input[1])
                negative = model(input[2])
                loss = triplet_loss(anchor, positive, negative)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                accumelated_loss += loss.item()

                batches_tbar.set_description(f'Batch {epoch_steps}/{len(train_dataset)} | Loss: {loss.item():.2f}')
                epochs_tbar.set_description(f'Epoch {epoch+1}/{num_epochs} | Average loss: {accumelated_loss/epoch_steps:.2f} | Accuracy: {accuracy:.2f}')
                batches_tbar.refresh()
                epochs_tbar.refresh()
                wandb.log({'loss': loss.item()})

                if (steps % eval_every == 0) or (epoch_steps == len(train_dataset)):
                    print('Evaluating model')
                    #Calculate triplet loss on eval dataset
                    model.eval()
                    eval_loss = 0
                    eval_steps = 0
                    for input in tqdm(eval_dataset, unit='batch', desc='Calculating triplet loss on eval dataset'):
                        anchor = model(input[0])
                        positive = model(input[1])
                        negative = model(input[2])
                        loss = triplet_loss(anchor, positive, negative)
                        eval_loss += loss.item()
                        eval_steps += 1
                    eval_loss = eval_loss/eval_steps


                    embeddings = []
                    sentences = val_single_df['sentence'].tolist()
                    labels = val_single_df['group']
                    for sentence in tqdm(sentences, unit='sentence', desc='Generating embeddings'):
                        embedding = model(sentence).detach().cpu().numpy()
                        embeddings.append(embedding)
                    embeddings = np.array(embeddings).squeeze()
                    all_res = calculate_dsiatances_from_embeddings(embeddings, labels)
                    average_inner_distance  = all_res['average_inner_distance']
                    average_across_distance =all_res['average_across_distance']
                    accuracy = calculate_accuracy_from_embeddings(embeddings, labels)
                    
                    wandb.log({'eval_loss': eval_loss})
                    wandb.log({'average_inner_distance': average_inner_distance})
                    wandb.log({'average_across_distance': average_across_distance})
                    wandb.log({'accuracy': accuracy})
                    
                    model.train()

                if (steps % save_model_every == 0) or (epoch_steps == len(train_dataset)):
                    print('Saving model')
                    # lora_save_path = f'./models/LoRa/lora_model_{short_model_name}_{steps}'
                    lora_save_path = f'{save_model_path}/lora_{model_save_name}_{steps}'
                    lora_model = model.Bert_representations
                    lora_model.save_pretrained(lora_save_path)
                    print('Pushing model to hub')
                    hub_model_name = f'LoRa_{model_save_name}'
                    lora_model.push_to_hub(hub_model_name)


                







if __name__ == '__main__':
    #check length of sys.argv
    if len(sys.argv) <= 1:
        print('No arguments were given, using default arguments')
        
        rank = 64
        peft_config = LoraConfig(inference_mode=False,
                    r=rank,
                    lora_alpha=rank*2,
                    lora_dropout=0.05,
                    target_modules=['value','query','key', 'dense']
                    )
        main({'peft_config': peft_config}, use_argparse=False)
    else:
        print('Arguments were given, using given arguments')
        main({}, use_argparse=True)




# Training Report
# 1. Training Dataset:
#     - We used our custom preprocessed dataset as previously mentioned. The dataset is in the triplet format (anchor, positive, negative) and a total of 7112 triplets were used for training. 
# The dataset has 14 different classes and each class is represented by 508 triplets. The average sentence length is 5.5 words and the maximum sentence length is 26 words.
# 2. Model:
#     - The base model used is MiniLM-L12-v1 This is a pretrained sentence-transformers model that maps sentences & paragraphs to a 384 dimensional dense vector space and can be used for tasks like clustering or semantic search.
# It was trained on a large and diverse dataset of over 1 billion training pairs using a contrastive loss function.
#     - The model was fine-tuned using Parameter Efficient fine-tuning (Peft) technique. Peft is a technique that allows for fine-tuning of large models with a small number of parameters. 
# Specifically we used the LoRa technique which is a Peft technique that allows for fine-tuning of large models by only fine-tuning a small set of Low Rank matrices inserted in different layers of the model.
# We used a rank of 64 and the target modules were: value, query, key, and dense layers. This resulted in a model with 5M trainble parameters, which is 13.5% of the original model parameters.
# 3. Training:
#     - The model was trained for 10 epochs with a batch size of 16. The optimizer used was AdamW with a learning rate of 1e-5. The triplet loss function was used with a margin of 1.0 and a norm degree 2.
# The maximum sentence length was 200 words. The model was trained on a single GPU (Nvidia Tesla T4) and the training took 2 hours and 30 minutes.
