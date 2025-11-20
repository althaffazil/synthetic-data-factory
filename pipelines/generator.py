import os
import pandas as pd

from utils.download_model import download_model
from config.settings import MODEL_NAME, MODEL_FILE, NUM_SYNTH_PER_SEED

from langchain_community.llms import CTransformers
from langchain_core.prompts import PromptTemplate


# -----------------------------------------------------
# Load LLM (auto-download)
# -----------------------------------------------------
def load_llm():

    if not os.path.exists(MODEL_FILE):
        download_model()

    return CTransformers(
        model=MODEL_NAME,
        model_file=MODEL_FILE,
        max_new_tokens=256,
        temperature=0.9,
        gpu_layers=0
    )


# -----------------------------------------------------
# Prompt Builder
# -----------------------------------------------------
def get_prompt():
    template = """
    Generate ONE unique synthetic sentence for text classification.

    Requirements:
    - Reflect this label: {label}
    - Realistic, 8–20 words
    - Must NOT copy the example text

    Example (do NOT copy):
    "{example}"

    Return only the synthetic sentence.
    """

    return PromptTemplate(
        template=template,
        input_variables=["label", "example"]
    )


# -----------------------------------------------------
# Synthetic Data Generator (LCEL version)
# -----------------------------------------------------
def generate_synthetic(seed_df):
    llm = load_llm()
    prompt = get_prompt()

    # LCEL chain
    chain = prompt | llm

    generated = []

    for _, row in seed_df.iterrows():
        example = row["text"]
        label = row["label"]

        for _ in range(NUM_SYNTH_PER_SEED):
            out = chain.invoke({"label": label, "example": example})
            generated.append((out.strip(), label))

    return pd.DataFrame(generated, columns=["text", "label"])
