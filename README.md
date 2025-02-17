# ECE 209AS LLMs for Deepfake Detection

To get started, make a new Python environment. (Named "ece209as" here, but name it whatever you want):

```sh
conda create -n "ece209as" python=3.13.2 ipython
conda activate ece209as
```

Install the dependencies:

```
pip install -r requirements.txt
```

*Note: I'm not sure it matters, but `requirements.txt` uses PyTorch installed configured for Linux with CUDA 12.6. YMMV with the current requirements.txt. If it doesn't work, try `pip3 install torch torchvision torchaudio` instead.*

## Maybe Useful links:

* [Llama Recipes](https://github.com/huggingface/huggingface-llama-recipes)
* [Potential Dataset](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images)

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