# **Synthetic Data Factory**

A modular pipeline for generating high-quality synthetic text datasets using open-source LLMs. The system automates dataset creation, applies semantic similarity filtering for quality control, and runs fully locally with automatic model downloading.

---

## **⚙️ Installation**

### 1. Clone the repository

```bash
git clone https://github.com/your-username/repo-name.git
cd repo-name
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## **📥 Model Auto-Download**

The required GGUF model is downloaded automatically from HuggingFace Hub during the first run.
No manual steps needed.

---

## **▶️ Usage**

Run the project:

```bash
python main.py
```

The generated synthetic dataset will be saved to:

```
data/synthetic.csv
