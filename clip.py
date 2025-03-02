import os
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import clip
from transformers import CLIPProcessor, CLIPModel

from images_analysis import FAKE_IMAGE_DIR, REAL_IMAGE_DIR
from llm import get_device
from util import log

NUM_EPOCHS = 10


class ImageDataset(Dataset):
    def __init__(self, device, processor):
        self.real_images = [
            os.path.join(REAL_IMAGE_DIR, f) for f in os.listdir(REAL_IMAGE_DIR)
        ]
        self.fake_images = [
            os.path.join(FAKE_IMAGE_DIR, f) for f in os.listdir(FAKE_IMAGE_DIR)
        ]
        self.images = self.real_images + self.fake_images
        self.text_labels = ["a real image"] * len(self.real_images) + [
            "an AI-generated image"
        ] * len(self.fake_images)
        self.numeric_labels = [0] * len(self.real_images) + [1] * len(self.fake_images)
        self.device = device
        self.processor = processor

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert("RGB")
        image = self.processor(images=image, return_tensors="pt").pixel_values.squeeze(
            0
        )
        text = self.processor.tokenizer(
            self.text_labels[idx],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids.squeeze(0)
        numeric_label = self.numeric_labels[idx]

        return image, text, numeric_label


def convert_models_to_fp32(model):
    for param in model.parameters():
        param.data = param.data.float()
        if param.grad is not None:
            param.grad.data = param.grad.data.float()


def main():
    device = get_device()
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-5,
        weight_decay=0.2,
    )

    loss_img = nn.CrossEntropyLoss()
    loss_text = nn.CrossEntropyLoss()

    dataset = ImageDataset(device, processor)
    train_dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    log(f"Starting training for {NUM_EPOCHS} epochs")
    for epoch in range(NUM_EPOCHS):
        pbar = tqdm(train_dataloader, total=len(train_dataloader))
        correct = 0
        total = 0

        for batch in pbar:
            optimizer.zero_grad()

            images, texts, labels = batch
            images = images.to(device)
            texts = texts.to(device)
            labels = labels.to(device)

            outputs = model(input_ids=texts, pixel_values=images)
            logits_per_image = outputs.logits_per_image
            logits_per_text = outputs.logits_per_text

            ground_truth = torch.arange(len(images)).to(device)
            loss = (
                loss_img(logits_per_image, ground_truth)
                + loss_text(logits_per_text, ground_truth)
            ) / 2

            loss.backward()
            if device == "cuda":
                convert_models_to_fp32(model)
            optimizer.step()

            _, predicted = torch.max(logits_per_image, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = correct / total * 100
        pbar.set_description(
            f"Epoch {epoch + 1}/{NUM_EPOCHS} Loss: {loss.item():.4f} Accuracy: {accuracy:.2f}%"
        )
        print(f"Epoch {epoch + 1} Accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    main()
