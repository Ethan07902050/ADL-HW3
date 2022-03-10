from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
import numpy as np
import sys

model_name = 'google/mt5-small'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

max_input_length = 256
max_target_length = 64
prefix = "summarize: "
batch_size = 16


def preprocess_function(examples):
    inputs = [prefix + doc for doc in examples["maintext"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["title"], max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def train(data_path, model_path):
    datasets = load_dataset('json', data_files={'train': data_path})
    tokenized_datasets = datasets.map(preprocess_function, batched=True)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    args = Seq2SeqTrainingArguments(
        model_path,
        save_strategy='no',
        learning_rate=4e-5,
        per_device_train_batch_size=batch_size,
        weight_decay=0.001,  
        num_train_epochs=16,
        fp16=True,
    )
    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=tokenized_datasets["train"],        
        data_collator=data_collator,
        tokenizer=tokenizer,    
    )

    trainer.train()


if __name__ == '__main__':
    data_path = sys.argv[1]
    model_path = sys.argv[2]
    train(data_path, model_path)
