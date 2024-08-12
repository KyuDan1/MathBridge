import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, BartTokenizer, BartForConditionalGeneration, MBartForConditionalGeneration, MBartTokenizer
from datasets import load_metric, load_dataset
import pandas as pd
import openai

# Replace with your actual OpenAI API key
openai.api_type = "azure"
openai.api_base = "https://speechai.openai.azure.com/"
openai.api_version = "2024-02-15-preview"
openai.api_key = "254539758a4d4a139feded9b9ae1b057"


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

import time

def evaluate_gpt_3_5(model_name, source_texts, target_texts, use_prompt):
    predictions = []
    references = []
    for source_text, target in zip(source_texts, target_texts):
        if use_prompt:
            prompt = f"The following sentence mixes spoken parts of formulas with standard English. Translate the part of the sentence that represents a formula into LaTeX.\n\n{source_text}\n\n"
        else:
            prompt = source_text

        response = openai.ChatCompletion.create(
            engine="gpt1",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0.2
        )

        try:
            prediction = response.choices[0].message['content'].strip()
        except Exception as e:
            print(e)
            time.sleep(60)

        predictions.append(prediction)
        references.append(target[0])

    # Calculate metrics
    bleu_score = bleu.compute(predictions=[pred.split() for pred in predictions], references=[[ref.split()] for ref in references])
    sacrebleu_score = sacrebleu.compute(predictions=predictions, references=[[ref] for ref in references])
    rouge_score = rouge.compute(predictions=predictions, references=references)
    cer_score = cer.compute(predictions=predictions, references=references)
    wer_score = wer.compute(predictions=predictions, references=references)

    # Save results to a text file
    with open(f"gpt3_5_evaluation_results_{'with_prompt' if use_prompt else 'no_prompt'}.txt", 'w', encoding='utf-8') as file:
        file.write(f"Model: GPT-3.5 {'with prompt' if use_prompt else 'without prompt'}\n")
        file.write(f"BLEU: {bleu_score['bleu']}\n")
        file.write(f"sacreBLEU: {sacrebleu_score['score']}\n")
        file.write(f"ROUGE: {rouge_score['rouge1'].mid.fmeasure}\n")
        file.write(f"CER: {cer_score}\n")
        file.write(f"WER: {wer_score}\n")
        file.write("Predictions and References:\n")
        for pred, ref in zip(predictions, references):
            file.write(f"Prediction: {pred}, Reference: {ref}\n")


    return predictions, bleu_score, sacrebleu_score, rouge_score, cer_score, wer_score

# Evaluate both cases
print("Evaluating GPT-3.5 without prompt")
evaluate_gpt_3_5("gpt-3.5", source_texts, target_texts, use_prompt=False)

print("Evaluating GPT-3.5 with prompt")
evaluate_gpt_3_5("gpt-3.5", source_texts, target_texts, use_prompt=True)