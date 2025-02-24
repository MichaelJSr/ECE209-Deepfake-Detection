from images_analysis import (
    FAKE_EMBEDDINGS_DIR,
    REAL_EMBEDDINGS_DIR,
    init_models,
    load_images,
    process_images,
)
from util import log


def main():
    log("Initializing models...")
    processor, model, device = init_models()
    log("Models initialized.")

    log("Loading images...")
    real_images, fake_images = load_images()
    log("Images loaded.")

    process_images(
        real_images, processor, model, REAL_EMBEDDINGS_DIR, "Real Images", device
    )
    process_images(
        fake_images, processor, model, FAKE_EMBEDDINGS_DIR, "Fake Images", device
    )

    log("All embeddings saved.")


if __name__ == "__main__":
    main()
