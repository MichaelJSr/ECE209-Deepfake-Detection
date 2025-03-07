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

from llm import FAKE_TRAIN_DIR, REAL_TRAIN_DIR, FAKE_TEST_DIR, REAL_TEST_DIR, TOP_FEATURES, get_device, get_embeddings
from util import log

MODEL_NAME = "fine-tuned-distilbert"
TOKENIZER_NAME = "fine-tuned-distilbert-tokenizer"
SAVE_FREQUENCY = "no"


def embedding_to_text(embedding):
    # select the top features
    embedding = embedding[TOP_FEATURES]

    # normalize the features
    embedding = embedding / np.linalg.norm(embedding)

    return " ".join([str(x) for x in embedding])


def get_test_dataset():
    test_embeddings, test_labels = get_embeddings(REAL_TEST_DIR, FAKE_TEST_DIR)

    texts = [embedding_to_text(embedding) for embedding in test_embeddings]
    labels = [int(label) for label in test_labels]

    dataset = Dataset.from_dict({"text": texts, "label": labels})
    dataset = dataset.train_test_split(test_size=0.8)
    return dataset


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


def test_finetuned_model():
    device = get_device()

    tokenizer = DistilBertTokenizer.from_pretrained(TOKENIZER_NAME)
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME)
    model.to(device)

    test_dataset = get_test_dataset()
    tokenized_test_dataset = test_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
    )

    training_args = TrainingArguments(
        output_dir="./results",
        overwrite_output_dir=True,
        evaluation_strategy=SAVE_FREQUENCY,
        learning_rate=1e-6,
        per_device_train_batch_size=256,
        per_device_eval_batch_size=256,
        num_train_epochs=10000,
        weight_decay=0.01,
        max_grad_norm=1.0,
        label_smoothing_factor=0.1,
        save_strategy=SAVE_FREQUENCY,
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
        train_dataset=tokenized_test_dataset["train"],
        eval_dataset=tokenized_test_dataset["test"],
        tokenizer=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)] if SAVE_FREQUENCY != "no" else None,
    )

    predictions = trainer.predict(tokenized_test_dataset["test"])
    predicted_labels = np.argmax(predictions.predictions, axis=1)

    accuracy = accuracy_score(tokenized_test_dataset["test"]["label"], predicted_labels)
    log(f"Test Accuracy: {accuracy}")


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
        overwrite_output_dir=True,
        evaluation_strategy=SAVE_FREQUENCY,
        learning_rate=1e-6,
        per_device_train_batch_size=256,
        per_device_eval_batch_size=256,
        num_train_epochs=500,
        weight_decay=0.01,
        max_grad_norm=1.0,
        label_smoothing_factor=0.1,
        save_strategy=SAVE_FREQUENCY,
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
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)] if SAVE_FREQUENCY != "no" else None,
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
    test_finetuned_model()
