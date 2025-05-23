# 🧠 Fine-Tuning Mamba with Learnable Inputs (Diploma Research)

This repository contains the code and experiments for a diploma project focused on **parameter-efficient fine-tuning** of the [Mamba](https://arxiv.org/abs/2312.00752) model using **learnable input tokens**. The core idea is to adapt the **prefix-tuning** method to better align with the **recurrent nature** of Mamba, resulting in a new approach: **periodic-tuning**.

---

## 🔬 Research Focus

The project explores the limitations of classic prefix-tuning when applied to Mamba and proposes **periodic-tuning** as a more effective alternative. Unlike traditional prefix-tuning, which prepends a block of trainable tokens at the beginning of the sequence, periodic-tuning **inserts the same learnable tokens at fixed intervals** throughout the input. This helps retain task-relevant information over long contexts.

---

## 📁 Repository Structure

- `configs/` – YAML configuration files for training and evaluation setups.
- `models/` – Pretrained model checkpoints (`.pth` files) used in Induction Heads experiments.
- `notebooks/` – Research Jupyter notebooks with exploratory analysis, visualizations, and experiment logs.
- `src/`
  - `data/` – Code for datasets creation.
  - `models/` – Mamba model definition with additional classification head.
  - `training_functions.py` – Core training logic: token insertion, embedding optimization, evaluation routines, etc.
  - `train_*.py` – Entry-point scripts for launching training runs with various configurations.

---

## 🧪 Methods Compared

The following fine-tuning strategies are evaluated:

- ✅ Full fine-tuning  
- ✅ LoRA (Low-Rank Adaptation)  
- ✅ Classic Prefix-Tuning  
- ✅ **Proposed: Periodic-Tuning**

Each method is tested under:
- **Standard setting** – training and evaluation on similar-length inputs
- **Short vs Long setting** – training on short sequences, evaluating on longer ones

---

## 📈 Key Results

- **Periodic-tuning outperforms prefix-tuning** on long input sequences, especially in generalization scenarios.
- It shows interpretable behavior: inserted tokens **"remind" the model of the task**, boosting prediction confidence.

---

## ⚙️ Requirements

- Hugging Face `transformers`, `datasets`, `accelerate`
- `mamba-ssm`
- `triton`
- `wandb` (optional, for logging)
- etc

**Install dependencies**:

```bash
pip install -r requirements.txt
```

## 👤 Author

This repository is part of a diploma research project by Aleksei Vlasov, conducted in 2024–2025.