import pandas as pd
from sentence_transformers import util
from config.settings import SIMILARITY_THRESHOLD
from models.embedder import load_embedder

# -----------------------------------------------------
# Filter Synthetic Data Using Embeddings
# -----------------------------------------------------
def filter_synthetic(seed_df, synth_df, embedder_name):

    embedder = load_embedder(embedder_name)

    seed_emb = embedder.encode(seed_df["text"].tolist(), convert_to_tensor=True)
    synth_emb = embedder.encode(synth_df["text"].tolist(), convert_to_tensor=True)

    similarity = util.cos_sim(synth_emb, seed_emb)
    mask = (similarity.max(dim=1).values < SIMILARITY_THRESHOLD).cpu().numpy()

    return synth_df[mask].reset_index(drop=True)
