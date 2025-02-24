import torch
from tqdm import tqdm
from transformers import ViTImageProcessor, ViTModel
from PIL import Image
import numpy as np
from pathlib import Path

from util import log

REAL_IMAGE_DIR = "test/REAL"
FAKE_IMAGE_DIR = "test/FAKE"
REAL_EMBEDDINGS_DIR = "embeddings_test/REAL"
FAKE_EMBEDDINGS_DIR = "embeddings_test/FAKE"
IMAGE_BATCH_SIZE = 512


def init_models():
    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
    model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Using device: {device}")
    return processor, model, device


def load_images():
    """
    Load all images from 'train/REAL' and 'train/FAKE' directories,
    skipping those that already have corresponding embeddings.

    Returns:
        real_images (list): List of tuples (image path, PIL Image) for real images.
        fake_images (list): List of tuples (image path, PIL Image) for fake images.
    """
    real_dir = Path(REAL_IMAGE_DIR)
    fake_dir = Path(FAKE_IMAGE_DIR)

    real_embeddings_dir = Path(REAL_EMBEDDINGS_DIR)
    fake_embeddings_dir = Path(FAKE_EMBEDDINGS_DIR)

    def _load_helper(dir, embeddings_dir):
        images = []
        for img_path in dir.glob("*"):
            if img_path.is_file():
                embedding_path = embeddings_dir / f"{img_path.stem}.npy"
                if not embedding_path.exists():
                    try:
                        img = Image.open(img_path).convert("RGB")
                        images.append((img_path, img))
                    except Exception as e:
                        log(f"Skipping {img_path} due to error: {e}")
        return images

    real_images = _load_helper(real_dir, real_embeddings_dir)
    fake_images = _load_helper(fake_dir, fake_embeddings_dir)

    return real_images, fake_images


def generate_embeddings(
    img: Image, processor: ViTImageProcessor, model: ViTModel, device: torch.device
):
    """
    Generate an embedding for a given image using the ViT model.

    Args:
        img (PIL.Image): The input image.
        processor (ViTImageProcessor): Image processor for ViT.
        model (ViTModel): ViT model.
        device (torch.Device): the device (GPU or CPU) to run the model on.

    Returns:
        numpy.ndarray: The extracted embedding as a 1D NumPy array.
    """
    inputs = processor(images=img, return_tensors="pt").to(device)
    model.to(device)
    outputs = model(**inputs)

    last_hidden_states = outputs.last_hidden_state

    # output is 1x197x768, goal is to get 1D vector
    # for now, use global average pooling
    pooled_output = last_hidden_states.mean(dim=1)
    embedding = pooled_output.flatten().detach().cpu().numpy()

    return embedding


def save_embedding(embedding, output_dir, filename):
    """
    Save an individual embedding to a .npy file.

    Args:
        embedding (numpy.ndarray): The embedding vector.
        output_dir (Path): The directory to save the embedding.
        filename (str): The filename (without extension) to use for saving.
    """
    output_path = Path(output_dir) / f"{filename}.npy"
    np.save(output_path, embedding)


def process_images(images, processor, model, output_dir, label, device):
    log(f"Processing {len(images)} {label.lower()}...")

    for img_path, img in tqdm(
        images, desc=f"Processing {label}", unit="image", position=0, leave=True
    ):
        embedding = generate_embeddings(img, processor, model, device)
        save_embedding(embedding, output_dir, img_path.stem)

    log(f"{len(images)} {label.lower()} processed.")
