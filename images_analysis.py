import torch
from tqdm import tqdm
from transformers import ViTImageProcessor, ViTModel
from PIL import Image
import numpy as np
from pathlib import Path

from util import log

REAL_IMAGE_DIR = "test/REAL"
FAKE_IMAGE_DIR = "test/FAKE"
REAL_EMBEDDINGS_DIR = "embeddings/embeddings_test/REAL"
FAKE_EMBEDDINGS_DIR = "embeddings/embeddings_test/FAKE"
IMAGE_BATCH_SIZE = 512


def init_models() -> tuple[ViTImageProcessor, ViTModel, torch.device]:
    """
    Initialize the ViT models for generating embeddings.

    Returns:
        :processor (ViTImageProcessor): google/vit-base-patch16-224-in21k.
        :model (ViTModel): google/vit-base-patch16-224-in21k.
        :device (torch.device): Either cuda/mps, or cpu if neither of those are available.
    """
    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
    model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    except:
        log("Could not initialize cuda/mps device, defaulting to cpu.")
        device = torch.device("cpu")

    log(f"Using device: {device}")
    return processor, model, device


def load_images() -> tuple[list[tuple[Path, Image.Image]], list[tuple[Path, Image.Image]]]:
    """
    Load all images from 'train/REAL' and 'train/FAKE' directories,
    skipping those that already have corresponding embeddings.

    Returns:
        :real_images (list): List of tuples (image path, PIL Image) for real images.
        :fake_images (list): List of tuples (image path, PIL Image) for fake images.
    """
    real_dir = Path(REAL_IMAGE_DIR)
    fake_dir = Path(FAKE_IMAGE_DIR)

    real_embeddings_dir = Path(REAL_EMBEDDINGS_DIR)
    fake_embeddings_dir = Path(FAKE_EMBEDDINGS_DIR)

    def _load_helper(dir, embeddings_dir) -> list[tuple[Path, Image.Image]]:
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
) -> np.ndarray:
    """
    Generate an embedding for a given image using the ViT model.

    Args:
        :img (PIL.Image): The input image.
        :processor (ViTImageProcessor): Image processor for ViT.
        :model (ViTModel): ViT model.
        :device (torch.Device): the device (GPU or CPU) to run the model on.

    Returns:
        :numpy.ndarray: The extracted embedding as a 1D NumPy array.
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


def save_embedding(embedding: np.ndarray, output_dir: str, filename: str):
    """
    Save an individual embedding to a .npy file.

    Args:
        :embedding (numpy.ndarray): The embedding vector.
        :output_dir (Path): The directory to save the embedding.
        :filename (str): The filename (without extension) to use for saving.
    """
    output_path = Path(output_dir) / f"{filename}.npy"
    np.save(output_path, embedding)


def load_embedding(embed_path: str) -> np.ndarray:
    """
    Load an individual embedding into a numpy array.

    Args:
        :embed_path (Path): The full directory + filename path to load the embedding from.

    Returns:
        :numpy.ndarray: The extracted embedding as a 1D NumPy array.
    """
    return np.load(Path(embed_path))

def load_embeddings(embed_path: str) -> list[np.ndarray]:
    """
    Load all embeddings into a list of numpy arrays.
    """
    return [np.load(Path(embed_path)) for embed_path in Path(embed_path).glob("*.npy")]

def convert_embedding_to_str(embedding: np.ndarray) -> str:
    """
    Convert a numpy array into its string representation.

    Args:
        :embedding (np.ndarray): A numpy array to convert to a string.

    Returns:
        :str: The embedding as a string.
    """
    return np.array_str(embedding)


def process_images(images: tuple[tuple[Path, Image.Image]], processor: ViTImageProcessor, model: ViTModel, output_dir: str, label: str, device: torch.device):
    """
    The engine that generates embeddings from the input images.

    Args:
        :images: A list of images to generate embeddings of.
        :processor: The processor to utilize, such as ViTImageProcessor.
        :model: The model to utilize, such as ViTModel.
        :output_dir: The directory to output embeddings to.
        :label: A label such as "real" or "fake" to indicate the type of embedding.
        :device: The torch device, whether it's cuda, mps, or cpu.
    """
    log(f"Processing {len(images)} {label.lower()}...")

    for img_path, img in tqdm(
        images, desc=f"Processing {label}", unit="image", position=0, leave=True
    ):
        embedding = generate_embeddings(img, processor, model, device)
        save_embedding(embedding, output_dir, img_path.stem)

    log(f"{len(images)} {label.lower()} processed.")
