import numpy as np
from datasets import Dataset
from transformers import (
    DistilBertConfig,
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from sklearn.metrics import accuracy_score

from llm import FAKE_TRAIN_DIR, REAL_TRAIN_DIR, TOP_FEATURES, get_device, get_embeddings
from util import log

MODEL_NAME = "fine-tuned-distilbert"
TOKENIZER_NAME = "fine-tuned-distilbert-tokenizer"


def embedding_to_text(embedding):
    # select the top features
    embedding = embedding[TOP_FEATURES]

    # normalize the features
    embedding = embedding / np.linalg.norm(embedding)

    return " ".join([str(x) for x in embedding])


def get_dataset():
    train_embeddings, train_labels = get_embeddings(REAL_TRAIN_DIR, FAKE_TRAIN_DIR)

    texts = [embedding_to_text(embedding) for embedding in train_embeddings]
    labels = [int(label) for label in train_labels]

    dataset = Dataset.from_dict({"text": texts, "label": labels})
    dataset = dataset.train_test_split(test_size=0.2)
    return dataset


def get_model():
    model_name = "distilbert-base-uncased"

    config = DistilBertConfig.from_pretrained(model_name)
    config.dropout = 0.2
    config.attention_dropout = 0.2
    config.num_labels = 2

    tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    model = DistilBertForSequenceClassification.from_pretrained(
        model_name,
        config=config,
    )

    return model, tokenizer


def tokenize_function(examples, tokenizer):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


def main():
    device = get_device()

    model, tokenizer = get_model()
    model.to(device)

    train_dataset = get_dataset()
    tokenized_datasets = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
    )

    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=1e-5,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        num_train_epochs=6,
        weight_decay=0.01,
        max_grad_norm=1.0,
        label_smoothing_factor=0.1,
        save_strategy="epoch",
        logging_dir="./logs",
        logging_steps=10,
        fp16=True if device.type == "cuda" else False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        tokenizer=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )

    trainer.train()
    results = trainer.evaluate()
    log(f"Results:\n {results}")

    predictions = trainer.predict(tokenized_datasets["test"])
    predicted_labels = np.argmax(predictions.predictions, axis=1)
    accuracy = accuracy_score(tokenized_datasets["test"]["label"], predicted_labels)
    log(f"Accuracy: {accuracy}")

    model.save_pretrained(MODEL_NAME)
    tokenizer.save_pretrained(TOKENIZER_NAME)


if __name__ == "__main__":
    main()
