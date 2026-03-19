# Transformer Explainability and Adversarial Analysis for Emotion Classification


This project explores how a transformer-based emotion classification model behaves under different explainability methods and simple adversarial perturbations.

The core workflow is implemented in the notebook `main.ipynb`, which:
- loads a pretrained BERT model for emotion classification;
- computes predictions on a small set of test sentences;
- compares multiple white-box and black-box explanation methods;
- evaluates two character-level adversarial attacks focused on prediction changes and explanation drift.

The notebook uses the pretrained model `bhadresh-savani/bert-base-uncased-emotion` and relies on PyTorch, Transformers, Captum, SHAP, LIME, and TextAttack. :contentReference[oaicite:0]{index=0} :contentReference[oaicite:1]{index=1}

## Project structure

- `main.ipynb` — main notebook containing the full experimental pipeline.
- `requirements.txt` — Python dependencies required to run the notebook.

## Main goals

The notebook is organized into four main parts:

1. **Model loading and prediction setup**
   - Loads the tokenizer and sequence classification model.
   - Retrieves the model label set.
   - Defines helper forward functions and attribution utilities.
   - Runs predictions on a small manually selected set of emotionally diverse sentences. :contentReference[oaicite:4]{index=4}

2. **White-box explainability**
   - Layer Integrated Gradients (LIG)
   - Integrated Gradients (IG)
   - Gradient SHAP

   These methods use gradients or internal model representations to estimate token importance for the predicted class. Their outputs are later aggregated and compared sentence by sentence. :contentReference[oaicite:5]{index=5}

3. **Black-box explainability**
   - SHAP
   - LIME

   These methods treat the model as a black box and estimate token importance by perturbing the input and measuring how the prediction changes. The notebook compares the token scores produced by both methods. :contentReference[oaicite:6]{index=6}

4. **Adversarial attacks**
   - **Attack 1 — Prediction-changing attack**  
     Generates character-level perturbations and searches for candidates that flip the predicted label. Among successful candidates, it keeps the one with the highest confidence for the new class and compares the original and attacked LIME explanations.
   - **Attack 2 — Explanation-preserving attack**  
     Generates character-level perturbations that keep the original prediction unchanged while maximizing the change in the LIME explanation, under a confidence-drop constraint. :contentReference[oaicite:7]{index=7}

## Notebook components

### Model and utility functions
The notebook defines helper functions such as:
- `custom_forward`
- `forward_with_embeds`
- `summarize_attributions`
- `predict_proba`
- `predict_proba_lime`
- `predict_proba_texts`
- `get_prediction`
- `get_lime_token_scores` :contentReference[oaicite:8]{index=8}

### Explanation comparison utilities
To compare explanations across methods and across attacks, the notebook includes:
- top-token extraction utilities;
- Jaccard-based explanation drift;
- signed rank-shift comparison between token importance scores. :contentReference[oaicite:9]{index=9}

### Candidate generation
Adversarial candidates are generated with simple character-level substitutions using TextAttack transformations such as:
- homoglyph swaps;
- QWERTY-based substitutions;
- random character substitutions.

A small fallback mechanism is also included to ensure a minimum number of perturbed candidates. :contentReference[oaicite:10]{index=10}

## Installation

Create a virtual environment and install the dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Team

This project was developed as part of the **Machine Learning Security** course.

**Team Members:**

- Marras Francesco
- Melis Giulia
---
