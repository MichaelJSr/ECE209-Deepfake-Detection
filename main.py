from transformers import ViTImageProcessor, ViTModel
from os import walk
from os.path import join
from PIL import Image


def analyze_image(img):
    inputs = processor(images=img, return_tensors="pt")
    outputs = model(**inputs)

    last_hidden_states = outputs.last_hidden_state

    # output is 1x197x768, goal is to get 1D vector
    # for now, use global average pooling
    pooled_output = last_hidden_states.mean(dim=1)
    embedding = pooled_output.flatten()

    print(embedding)
    print(embedding.shape)

def init_models():
    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
    model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
    return processor, model

def load_images(n: int):
    '''
    Load n number real images and fake images.

    n: The number of images to load into real and fake image lists.
    '''
    real_images = []
    fake_images = []
    counter = n
    for subdir, _, files in walk("train/REAL"):
        for file in files:
            real_images.append(Image.open(join(subdir, file)))
            counter -= 1
            if not counter:
                break

    counter = n
    for subdir, _, files in walk("train/FAKE"):
        for file in files:
            fake_images.append(Image.open(join(subdir, file)))
            counter -= 1
            if not counter:
                break

    return real_images, fake_images


if __name__ == "__main__":
    processor, model = init_models()
    real_images, fake_images = load_images(100)
    analyze_image(real_images)
