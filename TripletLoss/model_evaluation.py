# Use the model to generate embeddings all sentences in our dataset and then plot them in 2D using PCA
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sentence_transformers import SentenceTransformer, util, InputExample
import pandas as pd
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
import os
import csv
import gzip


def get_sent_embeddings(model, sentences):
    return model.encode(sentences, show_progress_bar=True)

def get_node_name(id):
    data_path = './TripletLoss/dataset/data.csv'
    df = pd.read_csv(data_path)
    df_labels = df[df['node_name'].notnull()][['id', 'node_name']]
    df_labels.fillna(7.0, inplace=True)
    return df_labels[df_labels['id'] == id]['node_name'].values[0]

def get_vis_data():
    data_path = './TripletLoss/dataset/data.csv'
    df = pd.read_csv(data_path)    
    df.dropna(subset=['question'], inplace=True)
    df.dropna(subset=['id'], inplace=True)
    ids_with_count_greater_than_16 = df.id.value_counts()[df.id.value_counts() > 16].index.tolist()
    df = df[df['id'].isin(ids_with_count_greater_than_16)]
    questions = df['question'].tolist()
    ids = df['id'].tolist()
    node_names = [get_node_name(id) for id in ids]

    return questions, ids, node_names

def get_2d_embeddings(model_path, questions):
    model = SentenceTransformer(model_path)
    embeddings = get_sent_embeddings(model, questions)
    embeddings_2d = TSNE(n_components=2).fit_transform(embeddings)
    return embeddings_2d



def compare_sactter_plots(embeddings_2d_1, embeddings_2d_2, ids, title1= 'tuned model', title2 = 'bare model', cmap_name='tab10'):
    unique_ids = set(ids)
    colors = plt.cm.get_cmap(cmap_name, len(unique_ids))
    id_color_map = {id: colors(i) for i, id in enumerate(unique_ids)}
    # Visualize the embeddings colored by their ids with a legend of node names
    fig, ax = plt.subplots(2, figsize=(8, 10))

    scatter1 = ax[0].scatter(embeddings_2d_1[:, 0], embeddings_2d_1[:, 1], c=ids, cmap=cmap_name)
    ax[0].set_title(title1)

    scatter2 = ax[1].scatter(embeddings_2d_2[:, 0], embeddings_2d_2[:, 1], c=ids, cmap=cmap_name)
    ax[1].set_title(title2)

    legend_labels = [plt.Line2D([], [], marker='o', color=id_color_map[id], markersize=5, label=get_node_name(id)) for id in unique_ids]
    fig.legend(handles=legend_labels, loc='center', bbox_to_anchor=(0.5, 1.05), ncol=3)
    plt.tight_layout()
    plt.show()



def show_comparison_plot():
    bare_model_name = 'all-MiniLM-L6-v2'
    tuned_model_name = './TripletLoss/models/sbert_model'

    questions, ids, node_names = get_vis_data()

    bare_embeddings_2d = get_2d_embeddings(bare_model_name, questions)
    tuned_embeddings_2d = get_2d_embeddings(tuned_model_name, questions)

    compare_sactter_plots(bare_embeddings_2d, tuned_embeddings_2d, ids, title1= 'bare model', title2 = 'tuned model', cmap_name='tab10')


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


def print_sts_benchmark_scores():
    test_samples = get_sts_benchmark_data()
    test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, batch_size=16, name='sts-test')


    
    bare_model_name = 'all-MiniLM-L6-v2'
    tuned_model_name = './TripletLoss/models/sbert_model'

    bare_model = SentenceTransformer(bare_model_name)
    tuned_model = SentenceTransformer(tuned_model_name)

    print(f'{"="*10} {bare_model_name} Bare Model Reseluts {"="*10}')
    print(test_evaluator(bare_model))

    print(f'{"="*10} {tuned_model_name} Tuned Model Reseluts {"="*10}')
    print(test_evaluator(tuned_model))

if __name__ == '__main__':
    show_comparison_plot()

    # print_sts_benchmark_scores()