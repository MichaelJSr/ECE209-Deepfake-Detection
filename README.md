# ECE 209AS LLMs for Deepfake Detection

## Tasks

- [x] load data from the embeddings (it's all saved in npy files) and convert to string
- [x] shove as much as we can into system prompt (we have way more embeddings than will reasonably fit in the context window, prob a good idea to maybe do a random forest vote type classification provided the LLM doesn't take too long)
- [x] make embeddings for test set (didn't do this yet..., maybe will try later)
- [x] ask it to classify (maybe a couple times)
- [x] parse the result (shouldn't be too bad provided LLM follows the format)
- [x] pray that it's better than 50/50 (it's probably not)
    - Final Result: Correct: 16, Incorrect: 29, Accuracy: 0.35555555555555557

To get started, make a new Python environment. (Named "ece209as" here, but name it whatever you want):

```sh
conda create -n "ece209as" python=3.13.2 ipython
conda activate ece209as
```

Install the dependencies:

```
pip install -r requirements.txt
```

*Note: if you have a Nvidia GPU, you might want to also `pip install -r requirements2.txt` which includes CUDA supported versions of PyTorch*

You should also download and unzip the `embeddings.zip` file to get all of the image embeddings. You can get it from [Google Drive](https://drive.google.com/file/d/1HBwsjsHBtqj0Ynzq_PKeQiKjKNLwcZA-/view?usp=sharing). The organization is as follows:
```
embeddings/
    embeddings_test/
        FAKE/ <-- Embeddings for AI generated images
        REAL/ <-- embeddings for real images
    embeddings_train/
        FAKE/
        REAL/
```

Embeddings were generated from images in the [Kaggle CIFAKE dataset](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images)

## Maybe Useful links:

* [Llama Recipes](https://github.com/huggingface/huggingface-llama-recipes)
* [Potential Dataset](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images)
* [Existing Work](https://arxiv.org/abs/2403.14077)

## Current Models

*(picked mostly arbitrarily based on existing recipes)*

* Large Language Model: [meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)
* Vision Transformer: [google/vit-base-patch16-224-in21k](https://huggingface.co/google/vit-base-patch16-224-in21k)

## Procedure

*(This is an idea of what could be the minimum implementation. In-Context learning is probably simpler---although we might face some limitations due to maximum model context size---so we'll start with that. Time permitting, we can try fine-tuning.)*

1. Load a labeled dataset of AI-generated vs. real images.
2. Use a Vision Transformer to generate 1-dimensional image embeddings for those images.
3. Create a system prompt with a bunch of examples (as many as we can fit).
    * *This could also be done in a user prompt, but that would probably blow up context size even more...*
4. Give a user prompt with a new image embedding and pray that it classifies it.

### Example System/Task Prompt

*Inspired from the [LICO Paper](https://arxiv.org/pdf/2406.18851)*
```
Each x is an image and each y is a label if the image is AI-generated, labeled 'yes', or not, labeled 'no'.
Predict y given x.

x: <image embedding>, y: <yes|no>
... (many repetitions for in-context learning)
x: <image embedding>, y: <yes|no>
```

### Example User/Query Prompt

```
x: <new image embedding>, y:
```

## Results

* LLM In-Context Learning
    - Final Result: Correct: 16, Incorrect: 29, Accuracy: 0.35555555555555557
* DistilBert Encoder-Only Fine Tuning
* CLIP Multimodal Image-Text Pair Prediction
```
[2025-03-06 19:32:22] Starting training for 10 epochs
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2500/2500 [11:19<00:00,  3.68it/s]
Epoch 1 Accuracy: 39.19%
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2500/2500 [11:12<00:00,  3.72it/s]
Epoch 2 Accuracy: 39.07%
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2500/2500 [11:11<00:00,  3.72it/s]
Epoch 3 Accuracy: 38.33%
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2500/2500 [11:13<00:00,  3.71it/s]
Epoch 4 Accuracy: 39.68%
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2500/2500 [11:15<00:00,  3.70it/s]
Epoch 5 Accuracy: 39.20%
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2500/2500 [11:20<00:00,  3.68it/s]
Epoch 6 Accuracy: 40.73%
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2500/2500 [11:22<00:00,  3.66it/s]
Epoch 7 Accuracy: 39.95%
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2500/2500 [11:16<00:00,  3.70it/s]
Epoch 8 Accuracy: 39.51%
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2500/2500 [11:15<00:00,  3.70it/s]
Epoch 9 Accuracy: 39.44%
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2500/2500 [11:16<00:00,  3.69it/s]
Epoch 10 Accuracy: 39.62%
```