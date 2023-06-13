import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import argparse

###

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_paraphrases(input_text, num_sentences, output_file):
    set_seed(42)

    model = T5ForConditionalGeneration.from_pretrained('ramsrigouthamg/t5_paraphraser')
    tokenizer = T5Tokenizer.from_pretrained('ramsrigouthamg/t5_paraphraser')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    model = model.to(device)

    with open(input_text, 'r') as file:
        sentences = file.readlines()

    paraphrased_sentences = []

    for sentence in sentences:
        sentence = sentence.strip()
        text = "paraphrase: " + sentence + " </s>"

        encoding = tokenizer.encode_plus(text, pad_to_max_length=True, return_tensors="pt")
        input_ids, attention_masks = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)

        beam_outputs = model.generate(
            input_ids=input_ids, attention_mask=attention_masks,
            do_sample=True,
            max_length=256,
            top_k=120,
            top_p=0.98,
            early_stopping=True,
            num_return_sequences=num_sentences
        )

        paraphrases = []
        for beam_output in beam_outputs:
            sent = tokenizer.decode(beam_output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            if sent.lower() != sentence.lower() and sent not in paraphrases:
                paraphrases.append(sent)

        paraphrased_sentences.append(paraphrases)

    with open(output_file, 'w') as file:
        for i, paraphrases in enumerate(paraphrased_sentences):
            file.write("Original Sentence: {}\n".format(sentences[i].strip()))
            file.write("Paraphrased Sentences:\n")
            for j, paraphrase in enumerate(paraphrases):
                file.write("{}: {}\n".format(j, paraphrase))
            file.write("\n")

    print("Paraphrases generated and stored in", output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Paraphrase Generator")
    parser.add_argument("-i", "--input", type=str, default="mT5_base_english2pcm.txt", required=True, help="Path to the input file")
    parser.add_argument("-n", "--num_sentences", type=int, default=3, help="Number of paraphrases to generate")
    parser.add_argument("-o", "--output", type=str, default="output.txt", help="Path to the output file")
    args = parser.parse_args()

    input_file = args.input
    num_sentences = args.num_sentences
    output_file = args.output

    generate_paraphrases(input_file, num_sentences, output_file)
