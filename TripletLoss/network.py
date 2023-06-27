import torch
import torch.nn as nn
from peft import get_peft_model, PeftConfig, PeftModel
from transformers import AutoTokenizer, AutoModel


class STS_model(nn.Module): 
    def __init__(self, model_path, device='cpu', pef_config=None, add_bert=True): 
        super(STS_model, self).__init__() 
        self.device = device
        self.add_bert = add_bert
        if add_bert:
            self.Bert_representations, self.tokenizer = get_Bert_representations_model(model_path, pef_config)
            self.Bert_representations.train(mode=True)
            self.Bert_representations.to(device)

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
        sentence_embeddings = self.mean_pooling(model_output, model_input['attention_mask'])
        return sentence_embeddings
    

def get_sts_model(model_path, device='cpu', pef_config=None):
    model = STS_model(model_path, device, pef_config)
    model = model.to(device)
    return model


def get_Bert_representations_model(model_path, pef_config=None):
    lower_case_model_path = model_path.lower()
    if 'lora' in lower_case_model_path:
        # get LoRa model and lora config from model_path
        print('Loading LoRa model From HuggingFace Hub...: ', model_path)
        config = PeftConfig.from_pretrained(model_path)
        bert = AutoModel.from_pretrained(config.base_model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
        bert = PeftModel.from_pretrained(bert, model_path)
        for name, param in bert.named_parameters():
            if 'lora' in name:
                param.requires_grad = True
        bert.print_trainable_parameters()
    elif pef_config is not None:
        print('Loading Peft model From HuggingFace Hub...: ', model_path)
        bert = AutoModel.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        bert = get_peft_model(bert, pef_config)
        bert.print_trainable_parameters()
    else:
        print('Loading Bert model From HuggingFace Hub...: ', model_path)
        bert = AutoModel.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)

    return bert, tokenizer

