import pandas as pd
from sentence_transformers import InputExample

def get_processed_input_examples(path = './TripletLoss/dataset/data.csv'):
    '''
    This function takes a csv file path and returns a list of InputExample objects.
    
    Parameters:
        path (str): The path of the csv file.

    Returns:
        data (list): A list of InputExample objects.
    '''
    df = pd.read_csv(path)

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

        
    print(f'Number of InputExamples: {len(data)}')
    print(f'Example of InputExample: {data[0].texts}')
    return data
