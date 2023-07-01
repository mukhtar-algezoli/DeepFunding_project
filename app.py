import time
import torch
import numpy as np
import streamlit as st
from peft import PeftModel
from TripletLoss.network import STS_model
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_distances
from TripletLoss.triplet_dataset import get_sentence_id_label_df
from TripletLoss.model_evaluation import  calculate_accuracy_from_embeddings

global DEVICE
# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DEVICE = 'cpu'

my_sts_model = STS_model('', add_bert=False, device=DEVICE)
if 'my_sts_model' not in st.session_state:
    st.session_state['my_sts_model'] = my_sts_model



def get_ds():
    data_path = './TripletLoss/dataset/data.csv'
    eval_data_df = get_sentence_id_label_df(data_path)
    sentences = eval_data_df['sentence']
    labels = eval_data_df['id']
    return sentences, labels

def get_models_paths_optins():
    lora_peft_paths =[
    'ammarnasr/LoRa_all-MiniLM-L12-v1',
    'ammarnasr/LoRa_LoRa_all-MiniLM-L12-v1_rank_8',
    # 'ammarnasr/LoRa_all-mpnet-base-v2'
    ]
    base_model_paths = [
        'sentence-transformers/all-MiniLM-L12-v1',
        # 'sentence-transformers/all-mpnet-base-v2'
        ]
    return lora_peft_paths, base_model_paths

def init_page():
    #configure streamlit app
    st.set_page_config(
        page_title="LoRa",
        page_icon="🧊",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title('LoRa')
    # Add Refresh button to sidebar
    if st.sidebar.button('Refresh'):
        st.write('Refreshed!')

    st.sidebar.title('Settings')


def init_session_states():
    st.sidebar.write(st.session_state)
    if 'base_model' not in st.session_state:
        st.session_state['base_model'] = None
    if 'final_model' not in st.session_state:
        st.session_state['final_model'] = None
    if 'tokenizer' not in st.session_state:
        st.session_state['tokenizer'] = None
    if 'embeddings' not in st.session_state:
        st.session_state['embeddings'] = None


def predict_group(target_embedding, embeddings, labels):
    unique_groups = labels.unique()
    avarage_group_embeddings = []
    for group in unique_groups:
        group_indices = labels[labels == group].index
        group_embeddings = embeddings[group_indices]
        avarage_group_embeddings.append(group_embeddings.mean(axis=0))
    avarage_group_embeddings = np.array(avarage_group_embeddings)
    distances = cosine_distances(target_embedding, avarage_group_embeddings)
    predicted_group = unique_groups[np.argmin(distances)]
    return predicted_group

def get_stored_model():
    final_model = st.session_state['final_model']
    my_sts_model = st.session_state['my_sts_model']
    tokenizer = st.session_state['tokenizer']
    my_sts_model.tokenizer = tokenizer
    my_sts_model.Bert_representations = final_model
    return my_sts_model

def get_all_embeddings(my_sts_model, sentences):
    start_time = time.time()
    embeddings = []
    st.info('Getting embeddings for all sentences in the dataset')
    progress_bar = st.progress(0)
    count = 0
    size = len(sentences)
    sentences_list = sentences.tolist()
    my_sts_model = my_sts_model.to(DEVICE)
    with torch.no_grad():
        with st.empty():
            for sentence in sentences_list:
                embedding = my_sts_model(sentence).detach().cpu().numpy()
                embeddings.append(embedding)
                count += 1
                progress_bar.progress(count/size)
                st.write(f'Elapsed Time: {time.time() - start_time} seconds')
    embeddings = np.array(embeddings).squeeze()
    st.session_state['embeddings'] = embeddings
    st.success(f'Embeddings loaded successfully in {time.time() - start_time} seconds')
    return embeddings


if __name__ == '__main__':
    init_page()
    embeddings = None
    sentences, labels = get_ds()
    unique_groups = labels.unique()
    lora_peft_paths, base_model_paths = get_models_paths_optins()
    init_session_states()


    st.subheader('Base Model')
    base_model_path = st.selectbox('Select the base model', base_model_paths)
    if st.button('Load Base Model'):
        start_time = time.time()
        with st.spinner('Loading Base Model...'):
            tokenizer = AutoTokenizer.from_pretrained(base_model_path)
            base_model = AutoModel.from_pretrained(base_model_path)
            st.session_state['base_model'] = base_model
            st.session_state['tokenizer'] = tokenizer
            st.success(f'Base Model Loaded Successfully in {time.time() - start_time} seconds')


    st.subheader('LoRa Model')
    lora_peft_path = st.selectbox('Select the LoRa model', lora_peft_paths)
    if st.button('Load LoRa Model'):
        start_time = time.time()
        with st.spinner('Loading LoRa Model...'):
            final_model = PeftModel.from_pretrained(st.session_state['base_model'], lora_peft_path)
            st.session_state['final_model'] = final_model
            embeddings = None
            st.session_state['embeddings'] = embeddings
            st.success(f'LoRa Model Loaded Successfully in {time.time() - start_time} seconds')


    st.subheader('Example of sentence Labels')
    examples = {}
    for label in labels.unique():
        examples[label] = sentences[labels==label].iloc[0]
    st.write(examples)


    st.subheader('Sentence Classification')
    sentence = st.text_input('Enter a sentence to classify')
    if st.button('Classify'):
        my_sts_model = get_stored_model()
        embeddings = st.session_state['embeddings']
        with st.spinner('Classifying...'):
            if embeddings is None:
                embeddings = get_all_embeddings(my_sts_model, sentences)
            else:
                st.info('Embeddings already loaded')
            sentence_embedding = my_sts_model(sentence).detach().cpu().numpy()
            predicted_group = predict_group(sentence_embedding, embeddings, labels)
            st.success(f'Predicted Group: {predicted_group}')
            st.write('Sentences from the predicted group:')
            st.write(sentences[labels==predicted_group].tolist())


    st.subheader('Accuracy')
    if st.button('Calculate Accuracy'):
        my_sts_model = get_stored_model()
        embeddings = st.session_state['embeddings']
        with st.spinner(f'Calculating Accuracy for {base_model_path} and {lora_peft_path}...'):
            if embeddings is None:
                embeddings = get_all_embeddings(my_sts_model, sentences)
            else:
                st.info('Embeddings already loaded')
            accuracy = calculate_accuracy_from_embeddings(embeddings, labels)
            st.success(f'Accuracy: {accuracy}')






