# import os
# work_dir = '/content/drive/MyDrive/DeepFund/'
# os.chdir(work_dir)
# os.chdir('DeepFunding_project/TripletLoss')

from sklearn.metrics.pairwise import cosine_distances
from triplet_dataset import get_dataset, get_sentence_id_label_df, TripletDataset
from network import get_sts_model
from tqdm.auto import tqdm
import numpy as np
from model_evaluation import  calculate_dsiatances_from_embeddings, calculate_accuracy_from_embeddings
from sklearn.manifold import TSNE
import torch
import os
import torch.nn as nn
from peft import LoraConfig
import argparse
import sys

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
        'model_save_name': None
    }
    for key in default_args_dict.keys():
        if key not in args_dict.keys():
            args_dict[key] = default_args_dict[key]

    #Pretty print args
    print('Arguments:')
    for key in args_dict.keys():
        print(f'{key}: {args_dict[key]}')


    train(**args_dict)

    

def train(model_path, data_path='./dataset/data.csv', device='cuda', peft_config=None,
          batch_size=16, lr=1e-5, triplet_loss=None, num_epochs=5, max_len=100,
          eval_every=100,save_model_every=1000, shuffle=True, eval_data_path=None,
          save_model_path='./models/LoRa', model_save_name=None):
    
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

    data_df = get_dataset(data_path)
    eval_data_df = get_sentence_id_label_df(eval_data_path)
    sentences = eval_data_df['sentence']
    labels = eval_data_df['id']


    print('Training model...')
    epochs_tbar = tqdm(range(num_epochs), unit='epoch')
    steps = 0
    accuracy = 0
    for epoch in epochs_tbar:
        train_dataset = TripletDataset(data_df, tokenizer=tokenizer, device=device, batch_size=batch_size, shuffle=shuffle, max_len=max_len)
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
                batches_tbar.refresh()

                epochs_tbar.set_description(f'Epoch {epoch+1}/{num_epochs} | Average loss: {accumelated_loss/epoch_steps:.2f} | Accuracy: {accuracy:.2f}')
                epochs_tbar.refresh()

                
                if (steps % eval_every == 0) or (epoch_steps == len(train_dataset)):
                    print('Evaluating model')
                    embeddings = []
                    for sentence in tqdm(sentences, unit='sentence', desc='Generating embeddings'):
                        embedding = model(sentence).detach().cpu().numpy()
                        embeddings.append(embedding)
                    embeddings = np.array(embeddings).squeeze()
                    all_res = calculate_dsiatances_from_embeddings(embeddings, labels)
                    accuracy = calculate_accuracy_from_embeddings(embeddings, labels)

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

