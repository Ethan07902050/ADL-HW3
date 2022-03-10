from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
import jsonlines, json
import torch
import os
import sys

max_input_length = 256
max_target_length = 64
device = 'cuda:0'

def predict(model_name):    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_size = 'left'
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model = model.to(device)
    
    batch_size = 32
    preds = []

    for i in tqdm(range(0, len(testset), batch_size)):
        if i+batch_size < len(testset):
            data = testset[i:i+batch_size]
        else:
            data = testset[i:len(testset)]
        
        text = [obj['maintext'] for obj in data]
        ids = [obj['id'] for obj in data]

        inputs = tokenizer(text, return_tensors='pt', max_length=max_input_length, truncation=True, padding=True)
        output_sequences = model.generate(
            input_ids=inputs['input_ids'].to(device),
            attention_mask=inputs['attention_mask'].to(device),
            max_length=max_target_length,
            num_beams=5,
            no_repeat_ngram_size=2,
            early_stopping=True
        )
        outputs = [tokenizer.decode(x, skip_special_tokens=True) for x in output_sequences]
        preds += [{'title': output, 'id': _id} for _id, output in zip(ids, outputs)]

    return preds


if __name__ == '__main__':
    with open(sys.argv[1]) as f:
        testset = [json.loads(line) for line in f]

    model_name = './model'
    preds = predict(model_name)
    
    with jsonlines.open(sys.argv[2], mode='w') as writer:
        writer.write_all(preds)
    
