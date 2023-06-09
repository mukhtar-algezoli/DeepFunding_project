import pandas as pd
from torch.utils.data import DataLoader

from sentence_transformers import InputExample
from sentence_transformers import SentenceTransformer, InputExample, losses
# Read CSV file
df = pd.read_csv('/home/muhammed-saeed/DeepFunding_project/TripletLoss/dataset/data.csv')

# Group data by 'id'
grouped = df.groupby('id')

# Hold the InputExamples
data = []

# Iterate over each group
for name, group in grouped:
    questions = group['question'].tolist()

    # Prepare anchor-positive pairs from the same group
    for i in range(len(questions)-1):
        anchor = questions[i]
        positive = questions[i+1]

        # For each anchor-positive pair, prepare negative from a different group
        for other_name, other_group in grouped:
            if name != other_name:
                negative = other_group['question'].iloc[0]
                data.append(InputExample(texts=[anchor, positive, negative]))
                break

# Now 'data' list contains InputExamples with anchor and positive from same 'id' and negative from a different 'id'

# Model to be fine-tuned
model = SentenceTransformer('all-MiniLM-L6-v2')

# Define the dataloader
train_dataloader = DataLoader(data, shuffle=True, batch_size=16)
train_loss = losses.MultipleNegativesRankingLoss(model=model)

# Train the model
model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=10, warmup_steps=100)

# After training, the model can be used to compute sentence embeddings:
sentence_embeddings = model.encode(["This is an example sentence"])
print('sentence embeddings')
