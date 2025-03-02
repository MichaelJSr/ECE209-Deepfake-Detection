from datetime import datetime
import json
import os
import numpy as np
import torch
from transformers import pipeline

from images_analysis import load_embeddings
from util import log

FINAL_PROMPTS_DIR = "prompts"
REAL_TRAIN_DIR = "embeddings/embeddings_train/REAL"
FAKE_TRAIN_DIR = "embeddings/embeddings_train/FAKE"
REAL_TEST_DIR = "embeddings/embeddings_test/REAL"
FAKE_TEST_DIR = "embeddings/embeddings_test/FAKE"
MODEL = "meta-llama/Llama-3.2-3B-Instruct"
NUM_CONTEXT_EMBEDDINGS = 8
EMBEDDING_PRECISION = 10  # nums after the decimal (at most 20)
TOP_FEATURES = [
    448,
    664,
    702,
    281,
    274,
    512,
    572,
    302,
    250,
    499,
    99,
    55,
    311,
    401,
    63,
    328,
    668,
    139,
    130,
    387,
]
NUM_ATTEMPTS = 3
NUM_TESTS = 1000

SYSTEM_PROMPT = """
You are a helpful assistant to classify images as real or fake.
When responding, respond only with a single word: 'fake' or 'real'.
Each 'x' is an image embedding and 'y' is the label. Predict 'y' given 'x'.
"""


def get_device():
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    except:
        log("Could not initialize cuda/mps device, defaulting to cpu.")
        device = torch.device("cpu")
    return device


def get_embeddings(real_dir, fake_dir):
    # labels: real = 1, fake = 0
    log("Loading embeddings...")
    real_embeddings = np.array(load_embeddings(real_dir))
    real_labels = np.ones(len(real_embeddings))
    fake_embeddings = np.array(load_embeddings(fake_dir))
    fake_labels = np.zeros(len(fake_embeddings))

    all_embeddings = np.vstack((real_embeddings, fake_embeddings))
    all_labels = np.concatenate((real_labels, fake_labels))

    sample_indices = np.random.choice(
        len(all_embeddings), NUM_CONTEXT_EMBEDDINGS, replace=False
    )

    log(f"Loaded embeddings from {real_dir} and {fake_dir}.")
    context_embeddings = all_embeddings[sample_indices]
    context_labels = all_labels[sample_indices]

    return context_embeddings, context_labels


def get_context_prompt(context_embeddings, context_labels):
    # convert embeddings to strings
    context_prompt = [
        f"x: {[(round(float(embedding[i]), EMBEDDING_PRECISION)) for i in TOP_FEATURES]} | y: {'real' if label == 1 else 'fake'}"
        for embedding, label in zip(context_embeddings, context_labels)
    ]

    return context_prompt


def get_test_prompt(test_embeddings, test_labels):
    # randomly sample 1 test embedding
    test_index = np.random.randint(len(test_embeddings))
    test_embedding = test_embeddings[test_index]
    test_label = "real" if test_labels[test_index] == 1 else "fake"

    test_prompt = f"x: {[(round(float(test_embedding[i]), EMBEDDING_PRECISION)) for i in TOP_FEATURES]} | y: "

    return test_prompt, test_label


def make_system_prompt(content):
    return {"role": "system", "content": content}


def make_user_prompt(content):
    return {"role": "user", "content": content}


def save_chat_prompt(chat, answer):
    full_prompt = ""
    for prompt in chat[:-1]:
        full_prompt += f"{prompt['content']}\n"

    timestamp = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    filename = f"{timestamp}_{answer}.txt"

    with open(os.path.join(FINAL_PROMPTS_DIR, filename), "w") as f:
        f.write(full_prompt.strip())


def run_test(device, context_embeddings, context_labels, test_embeddings, test_labels):
    test_prompt, test_label = get_test_prompt(test_embeddings, test_labels)

    prompt = []
    prompt.append(make_system_prompt(SYSTEM_PROMPT))

    log("Calling the LLM...")
    generator = pipeline(
        model=MODEL,
        device=device,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
    )

    count_real = 0
    count_fake = 0
    for i in range(NUM_ATTEMPTS):
        log(f"Attempt {i + 1}/{NUM_ATTEMPTS}")

        # randomly sample context each time
        context_prompt = get_context_prompt(context_embeddings, context_labels)
        context_prompt.append(test_prompt)
        prompt.append(make_user_prompt("\n".join(context_prompt)))

        generation = generator(
            prompt, do_sample=True, temperature=1, top_p=0.9, max_new_tokens=50
        )

        prompt.pop()

        chat = generation[0]["generated_text"]
        prediction = chat[-1]["content"].strip().split()[0].lower()
        if prediction == "real":
            count_real += 1
        elif prediction == "fake":
            count_fake += 1
        else:
            log(f"Invalid prediction: {prediction}")
        log(f"Prediction: {prediction}")
        save_chat_prompt(chat, test_label)

    log(f"Real: {count_real}, Fake: {count_fake}")
    prediction = "real" if count_real > count_fake else "fake"
    if prediction == test_label:
        log(f"Correct: predicted {prediction} for {test_label}")
        return True
    else:
        log(f"Incorrect: predicted {prediction} for {test_label}")
        return False


if __name__ == "__main__":
    device = get_device()

    context_embeddings, context_labels = get_embeddings(REAL_TRAIN_DIR, FAKE_TRAIN_DIR)

    test_embeddings, test_labels = get_embeddings(REAL_TEST_DIR, FAKE_TEST_DIR)

    correct = 0
    incorrect = 0
    try:
        for _ in range(NUM_TESTS):
            is_correct = run_test(
                device, context_embeddings, context_labels, test_embeddings, test_labels
            )
            if is_correct:
                correct += 1
            else:
                incorrect += 1
            print(f"Correct: {correct}, Incorrect: {incorrect}")
    except:
        pass
    finally:
        accuracy = correct / (correct + incorrect)
        log(f"Correct: {correct}, Incorrect: {incorrect}, Accuracy: {accuracy}")
