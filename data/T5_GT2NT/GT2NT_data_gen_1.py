import torch

from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    TrainingArguments,
    Trainer
)
from datasets import load_dataset
import pandas as pd
from datasets import Dataset, DatasetDict
import json

#dataset 재정의
dataset = pd.read_parquet('T5_allocate.parquet', engine='fastparquet')


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
path_1 = "Hyeonsieun/GTtoNT_addmoretoken_ver2"
tokenizer_1 = T5Tokenizer.from_pretrained(path_1)
model_1 = T5ForConditionalGeneration.from_pretrained(path_1)
model_1.to(device)
model_1.eval()

def do_correction_3(text, model, tokenizer):
    input_text = f"translate the LaTeX equation to a text pronouncing the formula: {text}"
    inputs = tokenizer.encode(
        input_text,
        return_tensors='pt',
        max_length=325,
        padding='max_length',
        truncation=True
    ).to(device)

    # Get correct sentence ids.
    corrected_ids = model.generate(
        inputs,
        max_length=325,
        num_beams=5, # `num_beams=1` indicated temperature sampling.
        early_stopping=True
    )

    # Decode.
    corrected_sentence = tokenizer.decode(
        corrected_ids[0],
        skip_special_tokens=False
    )
    return corrected_sentence

NT = []
for i in range(0, 1600000):
    TeX = '$' + df['TeX'][i] + '$'
    print(f"{i} input : {TeX}")
    GT_raw = do_correction_3(TeX, model_1, tokenizer_1)
    start_index = GT_raw.find("<pad>") + len("<pad>")
    end_index = GT_raw.find("</s>")
    NT_result = GT_raw[start_index:end_index].strip()
    NT_result = NT_result.replace("<unk>", "")
    print(f"{i} result : {NT_result}")
    NT.append(NT_result)
    if i % 50000 == 0:
        df_1 = pd.DataFrame(NT, columns=['SpNT'])
        df_1.to_csv(f"NT{i}.csv", index=False, encoding="utf-8-sig")

df_1 = pd.DataFrame(NT, columns=['SpNT'])

df_1.to_csv("NT1.csv", index=False, encoding="utf-8-sig")

print("=====================convert finish========================")
print(df_1.head(30))
