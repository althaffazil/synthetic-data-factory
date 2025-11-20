from sentence_transformers import SentenceTransformer

def load_embedder(model_name):
    return SentenceTransformer(model_name)
