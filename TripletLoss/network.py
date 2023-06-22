from transformers import AutoTokenizer, AutoModel, AutoConfig
import torch
import torch.nn as nn
from model_evaluation import get_vis_data, compare_sactter_plots
from sklearn.manifold import TSNE
from tqdm import tqdm
import numpy as np
from peft import get_peft_model, LoraConfig
from torch.utils.data import DataLoader
import os
from peft import PeftConfig, PeftModel

RANK = 2
PEFT_CONFIG = LoraConfig(inference_mode=False, 
              r=RANK, 
              lora_alpha=RANK*2, 
              lora_dropout=0.05,
              # target_modules=["q_lin","k_lin"]
              target_modules=['value','query']
              )


class MLPLayer(nn.Module):
    """
    Head for getting sentence representations over RoBERTa/BERT's CLS representation.
    """
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()
    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = self.activation(x)
        return x
    

class STS_model(nn.Module): 
    def __init__(self, model_path, device='cpu', pef_config=None, mean_pooling=False): 
      super(STS_model, self).__init__() 
      self.mean_pooling_flag = mean_pooling
      config = AutoConfig.from_pretrained(model_path, return_dict=True)
      self.device = device
      self.tokenizer = AutoTokenizer.from_pretrained(model_path)
      self.Bert_representations = AutoModel.from_pretrained(model_path)
      if pef_config is not None:
        self.Bert_representations = get_peft_model(self.Bert_representations, pef_config)
        self.Bert_representations.print_trainable_parameters()
      self.Bert_representations.to(device)
      if not self.mean_pooling_flag:
        self.MLP_layer = MLPLayer(config)
        self.MLP_layer.to(device)


    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    


        
    def forward(self, model_input): 
      # Tokenize sentences if input is a string or a list of strings
        if isinstance(model_input, str) or (isinstance(model_input, list) and isinstance(model_input[0], str)):
            model_input = self.tokenizer(model_input, padding=True, truncation=True, max_length=128, return_tensors="pt")
            model_input = model_input.to(self.device)


        model_output = self.Bert_representations(**model_input)
        
        # Get the representation of [CLS]
        if self.mean_pooling_flag:
            sentence_embeddings = self.mean_pooling(model_output, model_input['attention_mask'])
            return sentence_embeddings
        model_output = model_output.last_hidden_state[:, 0, :]
        model_output = self.MLP_layer(model_output)
        return model_output



def get_sts_model(model_path, device='cpu', pef_config=PEFT_CONFIG, mean_pooling=True):
    model = STS_model(model_path, device, pef_config, mean_pooling)
    model = model.to(device)
    return model