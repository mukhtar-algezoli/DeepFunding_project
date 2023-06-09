import pandas as pd
from sentence_transformers import InputExample

# Read CSV file
df = pd.read_csv('data.csv')

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
