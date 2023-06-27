import os
import csv
import gzip
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_distances
from sentence_transformers import SentenceTransformer, util, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator


def calculate_dsiatances_from_embeddings(embeddings, labels):
    group_distances = []
    total_distances = []
    all_res = {'group_id': [], 'inner_distance': [], 'across_distance': []}
    unique_groups = labels.unique()

    for group in unique_groups:
        # Calculate pairwise cosine distances within the group
        group_indices = labels[labels == group].index
        group_embeddings = embeddings[group_indices]
        group_distance = cosine_distances(group_embeddings).mean()
        group_distances.append(group_distance)
        
        # Calculate pairwise cosine distances across groups
        other_indices = labels[labels != group].index
        other_embeddings = embeddings[other_indices]
        total_distance = cosine_distances(group_embeddings, other_embeddings).mean()
        total_distances.append(total_distance)

        # Append the results to the dictionary
        all_res['group_id'].append(group)
        all_res['inner_distance'].append(group_distance)
        all_res['across_distance'].append(total_distance)

    # Calculate the average distances
    average_group_distance = sum(group_distances) / len(group_distances)
    average_total_distance = sum(total_distances) / len(total_distances)
    all_res['average_inner_distance'] = average_group_distance
    all_res['average_across_distance'] = average_total_distance

    print("Average inner_distance  within groups(The lower  the better):", average_group_distance)
    print("Average across_distance across groups(The higher the better):", average_total_distance)
    return all_res

def calculate_accuracy_from_embeddings(embeddings, labels):
    total = 0
    correct = 0
    unique_groups = labels.unique()
    avarage_group_embeddings = []
    for group in unique_groups:
        group_indices = labels[labels == group].index
        group_embeddings = embeddings[group_indices]
        avarage_group_embeddings.append(group_embeddings.mean(axis=0))
    avarage_group_embeddings = np.array(avarage_group_embeddings)
    for i, embedding in enumerate(embeddings):
        total += 1
        distances = cosine_distances([embedding], avarage_group_embeddings)
        if np.argmin(distances)+1 == labels[i]:
            correct += 1

    
    acc  = correct / total
    print(f'Accuracy: {acc*100:.2f}%')
    return acc*100
















def get_sent_embeddings(model, sentences):
    return model.encode(sentences, show_progress_bar=True)

def get_node_name(id, data_path = './dataset/data.csv'):
    df = pd.read_csv(data_path)
    df_labels = df[df['node_name'].notnull()][['id', 'node_name']]
    df_labels.fillna(7.0, inplace=True)
    return df_labels[df_labels['id'] == id]['node_name'].values[0]

def get_vis_data(data_path = './TripletLoss/dataset/data.csv'):
    df = pd.read_csv(data_path)    
    df.dropna(subset=['question'], inplace=True)
    df.dropna(subset=['id'], inplace=True)
    ids_with_count_greater_than_16 = df.id.value_counts()[df.id.value_counts() > 2].index.tolist()
    df = df[df['id'].isin(ids_with_count_greater_than_16)]
    questions = df['question'].tolist()
    ids = df['id'].tolist()
    node_names = [get_node_name(id, data_path=data_path) for id in ids]

    return questions, ids, node_names

def get_2d_embeddings(model_path, questions):
    model = SentenceTransformer(model_path)
    embeddings = get_sent_embeddings(model, questions)
    embeddings_2d = TSNE(n_components=2).fit_transform(embeddings)
    return embeddings_2d



def compare_sactter_plots(embeddings_2d_1, embeddings_2d_2, ids,save_fig_name=None,title1= 'bare model', title2 = 'tuned model', cmap_name='tab20', show=True):
    unique_ids = set(ids)
    colors = plt.cm.get_cmap(cmap_name, len(unique_ids))
    id_color_map = {id: colors(i) for i, id in enumerate(unique_ids)}
    # Visualize the embeddings colored by their ids with a legend of node names
    if embeddings_2d_2 is not None:
        fig, ax = plt.subplots(2, figsize=(8, 10))
    else:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax = [ax]

    scatter1 = ax[0].scatter(embeddings_2d_1[:, 0], embeddings_2d_1[:, 1], c=ids, cmap=cmap_name)
    ax[0].set_title(title1)

    if embeddings_2d_2 is not None:
        scatter2 = ax[1].scatter(embeddings_2d_2[:, 0], embeddings_2d_2[:, 1], c=ids, cmap=cmap_name)
        ax[1].set_title(title2)

    legend_labels = [plt.Line2D([], [], marker='o', color=id_color_map[id], markersize=5, label=get_node_name(id)) for id in unique_ids]
    fig.legend(handles=legend_labels, loc='center', bbox_to_anchor=(0.5, 1.05), ncol=3)
    plt.tight_layout()
    if show:
        plt.show()

    if save_fig_name is not None:
        plt.savefig(save_fig_name)



def show_comparison_plot(tuned_model_name='./TripletLoss/models/sbert_model', show=True):
    bare_model_name = 'all-MiniLM-L6-v2'

    questions, ids, node_names = get_vis_data()

    bare_embeddings_2d = get_2d_embeddings(bare_model_name, questions)
    tuned_embeddings_2d = get_2d_embeddings(tuned_model_name, questions)
    save_fig_name = f'./figs/{tuned_model_name.split("/")[-1]}.png'
    compare_sactter_plots(bare_embeddings_2d, tuned_embeddings_2d, ids,save_fig_name=save_fig_name, title1= 'bare model', title2 = 'tuned model', cmap_name='tab10', show=show)


def get_sts_benchmark_data():
    # Download dataset if needed
    sts_dataset_path = './TripletLoss/dataset/stsbenchmark.tsv.gz'
    if not os.path.exists(sts_dataset_path):
        util.http_get('https://sbert.net/datasets/stsbenchmark.tsv.gz', sts_dataset_path)
    test_samples = []
    with gzip.open(sts_dataset_path, 'rt', encoding='utf8') as fIn:
        reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
        for row in reader:
            if row['split'] == 'test':
                score = float(row['score']) / 5.0 #Normalize score to range 0 ... 1
                test_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=score))
    return test_samples


def print_sts_benchmark_scores(tuned_model_name='./TripletLoss/models/sbert_model', print_bare_model_scores=False):
    test_samples = get_sts_benchmark_data()
    x = [sample.texts for sample in test_samples]
    test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, batch_size=16, name='sts-test')


    if print_bare_model_scores:
        bare_model_name = 'all-MiniLM-L6-v2'
        bare_model = SentenceTransformer(bare_model_name)
        print(f'{"="*10} {bare_model_name} Bare Model Reseluts {"="*10}')
        bare_res = test_evaluator(bare_model)
        print(bare_res)



    tuned_model = SentenceTransformer(tuned_model_name)
    print(f'{"="*10} {tuned_model_name} Tuned Model Reseluts {"="*10}')
    res = test_evaluator(tuned_model)
    print(res)

    #append the results to a file
    file_name = 'sts_benchmark_scores.csv'


    if not os.path.exists(file_name):
        with open(file_name, 'w') as f:
            f.write('model_name,socre\n')
    
    with open(file_name, 'a') as f:
        f.write(f'{tuned_model_name},{res}\n')

    print(f'Saved the results to {file_name}')

def get_average_distances(model_name = 'all-MiniLM-L6-v2'):

    # Load the dataframe
    
    # Get the data
    data_path = './TripletLoss/dataset/data.csv'
    df = pd.read_csv(data_path)
    df = df[['question', 'id']]

    # drop rows with nan values
    df = df.dropna()

    #Reset the index
    df = df.reset_index(drop=True)

    # rename columns to sentences and labels
    df.columns = ['sentence', 'label']

    # Load the sentence transformer model
    model = SentenceTransformer(model_name)

    # Generate sentence embeddings
    embeddings = model.encode(df['sentence'].tolist(), show_progress_bar=True)


    # Calculate average distances within and across groups
    group_distances = []
    total_distances = []
    all_res = {
        'group_id': [],
        'group_distance': [],
        'total_distance': [],
        'model_name': []
    }
    unique_groups = df['label'].unique()

    for group in unique_groups:
        group_indices = df[df['label'] == group].index
        group_embeddings = embeddings[group_indices]
        
        # Calculate pairwise cosine distances within the group
        group_distance = cosine_distances(group_embeddings).mean()
        group_distances.append(group_distance)
        
        # Calculate pairwise cosine distances across groups
        other_indices = df[df['label'] != group].index
        other_embeddings = embeddings[other_indices]
        total_distance = cosine_distances(group_embeddings, other_embeddings).mean()
        total_distances.append(total_distance)

        all_res['group_id'].append(group)
        all_res['group_distance'].append(group_distance)
        all_res['total_distance'].append(total_distance)
        all_res['model_name'].append(model_name)

    # Calculate the average distances
    average_group_distance = sum(group_distances) / len(group_distances)
    average_total_distance = sum(total_distances) / len(total_distances)

    print("Average distance within groups:", average_group_distance)
    print("Average distance across groups:", average_total_distance)

    return average_group_distance, average_total_distance, all_res

def save_distances(tuned_model_name='./TripletLoss/models/sbert_model', print_bare_model_scores=False):

    if print_bare_model_scores:
        bare_model_name = 'all-MiniLM-L6-v2'
        print(f'Caculating average distances for {bare_model_name}')
        _, _, bare_res = get_average_distances(bare_model_name)
        print('='*10)
        bare_res_df = pd.DataFrame(bare_res)


    print(f'Caculating average distances for {tuned_model_name}')
    _, _, tuned_res = get_average_distances(tuned_model_name)
    tuned_res_df = pd.DataFrame(tuned_res)


    file_name = f'average_distances.csv'
    if not os.path.exists(file_name):
        #save dataframes to csv
        tuned_res_df.to_csv(file_name, index=False)
    
    else:
        #read the file and append the new results
        df = pd.read_csv(file_name)
        df = df.append(tuned_res_df, ignore_index=True)
        df.to_csv(file_name, index=False)

    print(f'Saved the results to {file_name}')


    