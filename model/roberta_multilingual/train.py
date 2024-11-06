import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset

model_ft = AutoModelForSequenceClassification.from_pretrained("FacebookAI/xlm-roberta-base")
tokenizer = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-base")

# Get jigsaw data
DATAPATH = "data/jigsaw/test.csv"
LABELPATH = "data/jigsaw/test_labels.csv"
df = pd.read_csv(DATAPATH)
label_df = pd.read_csv(LABELPATH)
df = df.merge(label_df, on="id")
df = df[df["toxic"] != -1]
dataset = Dataset.from_pandas(df)
dataset = dataset.train_test_split(test_size=0.1)

# Training and testing datasets
train_dataset = dataset['train']
test_dataset = dataset["test"]

def tokenize(batch):
    return tokenizer(batch["content"], padding=True, truncation=True, max_length=512)

train_dataset = train_dataset.map(tokenize, batched=True, batch_size=512)
test_dataset = test_dataset.map(tokenize, batched=True, batch_size=512)
train_dataset = train_dataset.remove_columns(["id", "content", "lang"])
test_dataset = test_dataset.remove_columns(["id", "content", "lang"])
train_dataset = train_dataset.rename_columns({"toxic": "label"})
test_dataset = test_dataset.rename_columns({"toxic": "label"})
# Set dataset format
train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# Freeze xlm-roberta backbone, only tune classifier
model_ft.roberta.requires_grad = False
model_ft.classifier.requires_grad = True

# TrainingArguments
repository_id = "/Users/slmin/Desktop/Research/CSCS/llm-pretrain-data-toxicity-removal/model/xlm-roberta/ckpts"

training_args = TrainingArguments(
    output_dir=repository_id,
    num_train_epochs=6,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    evaluation_strategy="epoch",
    logging_dir=f"{repository_id}/logs",
    logging_strategy="steps",
    logging_steps=500,
    learning_rate=5e-5,
    weight_decay=0.01,
    warmup_steps=500,
    save_strategy="epoch",
    load_best_model_at_end=True,
    save_total_limit=2,
    report_to="tensorboard",
)

# Trainer
trainer = Trainer(
    model=model_ft,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)
# Fine-tune the model
trainer.train()

# Evaluate the model
trainer.evaluate()

# Save the model
trainer.save_model()