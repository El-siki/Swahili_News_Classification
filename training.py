import os


# Split data
train_df, test_df = train_test_split(df, test_size=TEST_SIZE, random_state=SEED, stratify=df["label"])
hf_datasets = DatasetDict({
"train": Dataset.from_pandas(train_df),
"test": Dataset.from_pandas(test_df)
})


# Tokenizer & Model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
MODEL_NAME,
num_labels=num_labels,
id2label=id2label,
label2id=label2id
)


def preprocess_function(examples):
return tokenizer(examples["cleaned"], truncation=True, max_length=MAX_LENGTH)


hf_datasets = hf_datasets.map(preprocess_function, batched=True)
hf_datasets = hf_datasets.rename_column("label", "labels")
hf_datasets.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])


accuracy = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")


def compute_metrics(pred):
logits, labels = pred
preds = np.argmax(logits, axis=-1)
return {
"accuracy": accuracy.compute(predictions=preds, references=labels)["accuracy"],
"f1": f1_metric.compute(predictions=preds, references=labels, average="macro")["f1"],
}


# Training arguments
args = TrainingArguments(
output_dir=OUTPUT_DIR,
evaluation_strategy="epoch",
save_strategy="epoch",
learning_rate=LR,
per_device_train_batch_size=BATCH_SIZE,
per_device_eval_batch_size=BATCH_SIZE,
num_train_epochs=EPOCHS,
weight_decay=WEIGHT_DECAY,
load_best_model_at_end=True,
metric_for_best_model="f1",
)


collator = DataCollatorWithPadding(tokenizer)


trainer = Trainer(
model=model,
args=args,
train_dataset=hf_datasets["train"],
eval_dataset=hf_datasets["test"],
tokenizer=tokenizer,
data_collator=collator,
compute_metrics=compute_metrics,
)


print("Starting training...")
trainer.train()
trainer.save_model(OUTPUT_DIR)
print("Training complete.")
