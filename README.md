# CS224N: Natural Language Processing with Deep Learning
### Stanford University — Assignment Solutions

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange?logo=pytorch)
![License](https://img.shields.io/badge/License-Academic-lightgrey)

**Author:** Ahmed Maged &nbsp;|&nbsp; ✉️ [ahmedmaged2004zsc@gmail.com](mailto:ahmedmaged2004zsc@gmail.com) &nbsp;|&nbsp; 🔗 [linkedin.com/in/ahmed-maged-swe](https://www.linkedin.com/in/ahmed-maged-swe/)

---

## Overview

This repository contains my completed assignments for **CS224N: Natural Language Processing with Deep Learning** at Stanford University. The course covers foundational and state-of-the-art NLP techniques, ranging from classical word vector methods to modern transformer-based architectures.

> **Note:** All implementations are my own. Code shared here is for learning and reference purposes — please adhere to Stanford's Honor Code if you are enrolled in the course.

---

## Repository Structure

```
cs224n-assignments/
│
├── a1/                          # Assignment 1 – Exploring Word Vectors
│   ├── exploring_word_vectors.ipynb
│   ├── imgs/
│   └── README.md
│
├── a2/                          # Assignment 2 – word2vec
│   ├── word2vec.py
│   ├── sgd.py
│   └── README.md
│
├── a3/                          # Assignment 3 – Dependency Parsing
│   ├── parser_model.py
│   ├── parser_transitions.py
│   └── README.md
│
├── a4/                          # Assignment 4 – Neural Machine Translation
│   ├── model_embeddings.py
│   ├── nmt_model.py
│   └── README.md
│
└── requirements.txt
```

---

## Assignments

### Assignment 1 — Exploring Word Vectors

**Topics:** Co-occurrence matrices, SVD, GloVe, cosine similarity, word analogy, bias analysis

Explores two families of word vectors: count-based embeddings built from co-occurrence matrices and reduced with Truncated SVD, and prediction-based embeddings from pretrained GloVe. Includes analysis of word clustering, polysemy, synonyms/antonyms, analogical reasoning, and gender bias.

**Dataset:** [Stanford IMDB Large Movie Review Dataset](https://huggingface.co/datasets/stanfordnlp/imdb)
**Pretrained model:** `glove-wiki-gigaword-200` (400k tokens, 200-dimensional)

---

### Assignment 2 — Implementing word2vec

**Topics:** Skip-gram model, negative sampling, stochastic gradient descent

Implements the word2vec skip-gram model from scratch, including:
- Softmax and negative-sampling loss functions
- Gradients derived and implemented by hand
- SGD with optional gradient clipping

---

### Assignment 3 — Neural Dependency Parsing

**Topics:** Transition-based parsing, feed-forward networks, dropout

Builds a transition-based dependency parser using a feed-forward neural network. Implements the arc-standard transition system and trains the model to predict the correct transition (SHIFT / LEFT-ARC / RIGHT-ARC) at each step.

---

### Assignment 4 — Neural Machine Translation

**Topics:** Encoder-decoder RNNs, attention mechanism, beam search

Implements a seq2seq NMT model with multiplicative attention. Evaluates translation quality using BLEU score on a Cherokee–English dataset.

---

## Environment Setup

### Prerequisites

- Python ≥ 3.8
- pip or conda

### Installation

```bash
# Clone this repository
git clone https://github.com/<your-username>/cs224n-assignments.git
cd cs224n-assignments

# Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

### Core Dependencies

```
numpy
scipy
scikit-learn
matplotlib
torch
gensim
datasets
jupyter
```

---

## Running the Notebooks

```bash
# Launch Jupyter from the assignment directory
cd a1
jupyter notebook exploring_word_vectors.ipynb
```

The first run of Assignment 1 will download the GloVe model (~800 MB). Subsequent runs load from the local Gensim cache and take roughly 1–2 minutes.

---

## Key Concepts Covered

**Word Representations**  
Co-occurrence matrices capture global statistics but produce sparse, high-dimensional vectors. Dimensionality reduction via SVD recovers dense, lower-dimensional embeddings. Prediction-based methods like GloVe learn directly in a low-dimensional space by factoring the log co-occurrence ratio matrix, yielding richer semantic structure.

**Bias in Embeddings**  
Word embeddings inherit the biases present in their training corpora. Queries such as `career + woman − man` and `career + man − woman` reveal that GloVe systematically associates women's careers with domestic framing and men's careers with leadership. Mitigation strategies include hard/soft debiasing (Bolukbasi et al., 2016) and multi-class debiasing (Manzini et al., 2019).

**Analogical Reasoning**  
GloVe vectors support analogy queries via vector arithmetic: `x ≈ w + (g − m)` for the analogy `man : grandfather :: woman : x`. Performance varies — geometric relationships hold for common semantic patterns but degrade for rare or culturally specific ones.

---

## References

- Pennington, J., Socher, R., & Manning, C. D. (2014). [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/pubs/glove.pdf). *EMNLP*.
- Bolukbasi, T., Chang, K., Zou, J., Saligrama, V., & Kalai, A. (2016). [Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings](https://arxiv.org/pdf/1607.06520). *NeurIPS*.
- Manzini, T., Chong, L. Y., Black, A. W., & Tsvetkov, Y. (2019). [Black is to Criminal as Caucasian is to Police](https://aclanthology.org/N19-1062.pdf). *NAACL*.
- Jurafsky, D., & Martin, J. H. (2023). [Speech and Language Processing (3rd ed. draft)](https://web.stanford.edu/~jurafsky/slp3/).

---

## License

This repository is for educational purposes. Please review Stanford's [Honor Code](https://communitystandards.stanford.edu/policies-guidance/honor-code) before reusing any assignment code.
