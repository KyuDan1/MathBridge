import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, BartTokenizer, BartForConditionalGeneration
from transformers import T5ForConditionalGeneration, T5Tokenizer,BartForConditionalGeneration, BartTokenizer, MBartForConditionalGeneration, MBartTokenizer

from datasets import load_metric, load_dataset

# Check the number of available CUDA devices
if torch.cuda.is_available():
    device = torch.device("cuda:0")  # Use the first available GPU
else:
    device = torch.device("cpu")  # Fallback to CPU if no GPU is available

# Define the models and their tokenizers
model_names = {
    #"t5-small": T5ForConditionalGeneration.from_pretrained("t5-small").to(device),
    #"t5-base": T5ForConditionalGeneration.from_pretrained("t5-base").to(device),
    #"t5-large": T5ForConditionalGeneration.from_pretrained("t5-large").to(device),
    #"bart-base": BartForConditionalGeneration.from_pretrained("facebook/bart-base").to(device),
    #"bart-large": BartForConditionalGeneration.from_pretrained("facebook/bart-large").to(device),
    "mbart-large-50": MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50").to(device)
}
print("model loaded")
tokenizers = {
    #"t5-small": T5Tokenizer.from_pretrained("t5-small"),
    #"t5-base": T5Tokenizer.from_pretrained("t5-base"),
    #"t5-large": T5Tokenizer.from_pretrained("t5-large"),
    #"bart-base": BartTokenizer.from_pretrained("facebook/bart-base"),
    #"bart-large": BartTokenizer.from_pretrained("facebook/bart-large"),
    "mbart-large-50": MBartTokenizer.from_pretrained("facebook/mbart-large-50")
}
print("tokenizer loaded")

# Load evaluation metrics
bleu = load_metric('bleu')
sacrebleu = load_metric('sacrebleu')  # Add sacreBLEU metric
rouge = load_metric('rouge')
cer = load_metric('cer')
wer = load_metric('wer')

print("metric loaded")

# Load your dataset
dataset = load_dataset('Kyudan/test_dataset', split='test')
source_texts = [before + " " + english + " " + after for before, english, after in zip(dataset['context_before'], dataset['spoken_English'], dataset['context_after'])]
target_texts = [[before + " " + equation + " "+ after] for before, equation, after in zip(dataset['context_before'], dataset['equation'], dataset['context_after'])]

print("test dataset loaded")

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

    return bleu_score, sacrebleu_score, rouge_score, cer_score, wer_score



# Evaluate all models
results = {}
for model_name, model in model_names.items():
    tokenizer = tokenizers[model_name]
    print(f"Evaluating {model_name}")
    bleu_score, sacrebleu_score, rouge_score, cer_score, wer_score = evaluate_model(model_name, model, tokenizer, source_texts, target_texts)
    results[model_name] = {
        "BLEU": bleu_score,
        "sacreBLEU": sacrebleu_score,
        "ROUGE": rouge_score,
        "CER": cer_score,
        "WER": wer_score
    }
    print(f"Results for {model_name}:\nBLEU: {bleu_score}\nsacreBLEU: {sacrebleu_score}\nROUGE: {rouge_score}\nCER: {cer_score}\nWER: {wer_score}\n")

    with open(f"{model_name}_results.txt", "w") as file:
        file.write(f"Results for {model_name}:\n")
        file.write(f"BLEU: {bleu_score}\n")
        file.write(f"sacreBLEU: {sacrebleu_score}\n")
        file.write(f"ROUGE: {rouge_score}\n")
        file.write(f"CER: {cer_score}\n")
        file.write(f"WER: {wer_score}\n")
