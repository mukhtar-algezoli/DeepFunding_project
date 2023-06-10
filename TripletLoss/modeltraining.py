import pandas as pd
from torch.utils.data import DataLoader
from sentence_transformers import InputExample
from sentence_transformers import SentenceTransformer, InputExample, losses
from datapreperation import get_processed_input_examples


def train_sbert():

    # Define the parameters
    data_path = './TripletLoss/dataset/data.csv'
    model_name = 'all-MiniLM-L6-v2'
    batch_size = 16
    epochs = 10
    warmup_steps = 100
    model_save_path = './TripletLoss/models/sbert_model'

    # Get the data
    data = get_processed_input_examples(data_path)
    new_data = []
    for i in range(len(data)):
        sample = data[i]
        #remove any sample where all three  samples.texts are not type(str)
        if not all(isinstance(text, str) for text in sample.texts):
            print(f'Found a sample with non-string text: {sample.texts}')
            continue
        new_data.append(sample)
    data = new_data

    # Model to be fine-tuned
    model = SentenceTransformer(model_name)

    # Define the dataloader
    train_dataloader = DataLoader(data, shuffle=True, batch_size=batch_size)
    train_loss = losses.MultipleNegativesRankingLoss(model=model)

    # Train the model
    model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=epochs, warmup_steps=warmup_steps)

    # Save the model
    model.save(model_save_path)



if __name__ == '__main__':
    train_sbert()