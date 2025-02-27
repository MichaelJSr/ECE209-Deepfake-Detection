from pathlib import Path

from images_analysis import (
    FAKE_EMBEDDINGS_DIR,
    REAL_EMBEDDINGS_DIR,
    load_embedding,
    convert_embedding_to_str
)
from util import log

NUM_EMBEDDINGS_TO_ANALYZE = 10


def main():
    def _load_helper(embeddings_dir, num_to_analyze):
        embeddings = []
        c = 0
        for embed_path in embeddings_dir.glob("*"):
            if embed_path.is_file():
                embeddings.append(load_embedding(embed_path))
                c += 1
            if c == num_to_analyze:
                break
        return embeddings

    log("Loading fake embeddings...")
    fake_embeddings = _load_helper(Path(FAKE_EMBEDDINGS_DIR), NUM_EMBEDDINGS_TO_ANALYZE)
    log("Fake embeddings loaded.\nLoading real embeddings...")
    real_embeddings = _load_helper(Path(REAL_EMBEDDINGS_DIR), NUM_EMBEDDINGS_TO_ANALYZE)
    log("Real embeddings loaded.")

    fake_embedding_strs = []
    for fake_embedding in fake_embeddings:
        fake_embedding_strs.append(convert_embedding_to_str(fake_embedding))
    del fake_embeddings

    real_embedding_strs = []
    for real_embedding in real_embeddings:
        real_embedding_strs.append(convert_embedding_to_str(real_embedding))
    del real_embeddings

if __name__ == "__main__":
    main()
