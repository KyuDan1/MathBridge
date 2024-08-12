import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, BartTokenizer, BartForConditionalGeneration, MBartForConditionalGeneration, MBartTokenizer
from datasets import load_metric, load_dataset
import pandas as pd

# Check the number of available CUDA devices
if torch.cuda.is_available():
    device = torch.device("cuda:0")  # Use the first available GPU
else:
    device = torch.device("cpu")  # Fallback to CPU if no GPU is available

# Define the models and their tokenizers
model_names = {
    "t5-small": T5ForConditionalGeneration.from_pretrained("t5-small").to(device),
    "t5-base": T5ForConditionalGeneration.from_pretrained("t5-base").to(device),
    "t5-large": T5ForConditionalGeneration.from_pretrained("t5-large").to(device),
    "bart-base": BartForConditionalGeneration.from_pretrained("facebook/bart-base").to(device),
    "bart-large": BartForConditionalGeneration.from_pretrained("facebook/bart-large").to(device),
    "mbart-large-50": MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50").to(device)
}
print("Models loaded")
tokenizers = {
    "t5-small": T5Tokenizer.from_pretrained("t5-small"),
    "t5-base": T5Tokenizer.from_pretrained("t5-base"),
    "t5-large": T5Tokenizer.from_pretrained("t5-large"),
    "bart-base": BartTokenizer.from_pretrained("facebook/bart-base"),
    "bart-large": BartTokenizer.from_pretrained("facebook/bart-large"),
    "mbart-large-50": MBartTokenizer.from_pretrained("facebook/mbart-large-50")
}
print("Tokenizers loaded")

# Load evaluation metrics
bleu = load_metric('bleu')
sacrebleu = load_metric('sacrebleu')  # Add sacreBLEU metric
rouge = load_metric('rouge')
cer = load_metric('cer')
wer = load_metric('wer')

print("Metrics loaded")

# Load your dataset
dataset = load_dataset('Kyudan/test_dataset', split='test')
source_texts = [before + " " + english + " " + after for before, english, after in zip(dataset['context_before'], dataset['spoken_English'], dataset['context_after'])]
target_texts = [[before + " " + equation + " "+ after] for before, equation, after in zip(dataset['context_before'], dataset['equation'], dataset['context_after'])]

print("Test dataset loaded")

def evaluate_model(model_name, model, tokenizer, source_texts, target_texts):
    model.eval()
    predictions = []
    references = []  # Prepare reference format for BLEU
    max_length = 512 
    with torch.no_grad():
        for source_text, target in zip(source_texts, target_texts):
            inputs = tokenizer(source_text, return_tensors='pt', padding=True, truncation=True, max_length=max_length).to(device)
            inputs = {k: v.to(device) for k, v in inputs.items()}  # Move inputs to the same device
            outputs = model.generate(**inputs, max_new_tokens=100)
            prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
            predictions.append(prediction)  # For sacreBLEU, predictions should not be tokenized
            references.append(target[0])  # For sacreBLEU, references should not be tokenized

    # Calculate metrics
    bleu_score = bleu.compute(predictions=[pred.split() for pred in predictions], references=[[ref.split()] for ref in references])
    sacrebleu_score = sacrebleu.compute(predictions=predictions, references=[[ref] for ref in references])  # Fix applied here
    rouge_score = rouge.compute(predictions=predictions, references=references)
    cer_score = cer.compute(predictions=predictions, references=references)
    wer_score = wer.compute(predictions=predictions, references=references)

    torch.cuda.empty_cache()  # Clear GPU memory

    return predictions, bleu_score, sacrebleu_score, rouge_score, cer_score, wer_score

# Evaluate all models and create a DataFrame
results = []
for model_name, model in model_names.items():
    tokenizer = tokenizers[model_name]
    print(f"Evaluating {model_name}")
    predictions, bleu_score, sacrebleu_score, rouge_score, cer_score, wer_score = evaluate_model(model_name, model, tokenizer, source_texts, target_texts)
    
    # Create a DataFrame for each model's results
    model_results = {
        "Model": [model_name] * len(predictions),
        "Source Text": source_texts,
        "Target Text": [ref[0] for ref in target_texts],
        "Predicted Text": predictions,
        "BLEU": [bleu_score['bleu']] * len(predictions),
        "sacreBLEU": [sacrebleu_score['score']] * len(predictions),
        "ROUGE": [rouge_score['rouge1'].mid.fmeasure] * len(predictions),
        "CER": [cer_score] * len(predictions),
        "WER": [wer_score] * len(predictions)
    }
    df_model_results = pd.DataFrame(model_results)
    results.append(df_model_results)
    print(f"Results for {model_name}:\n{df_model_results.head()}\n")

# Concatenate all results into a single DataFrame
df_results = pd.concat(results, ignore_index=True)

# Save the DataFrame to a CSV file
df_results.to_csv("model_evaluation_results.csv", index=False)
print("Results saved to model_evaluation_results.csv")
