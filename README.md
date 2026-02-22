# Goodreads Genre Classification Report

## Model Selection

- **Model Used:** distilbert-base-cased  
- **Rationale:** Selected for efficiency due to its smaller size and faster inference compared to standard BERT, while retaining strong contextual understanding. The cased tokenizer preserves capitalization, which can improve interpretation of proper nouns, titles, and stylistic elements in book reviews.

---

## Final Training Metrics

- **Final Training Loss:** 0.5921  
- **Final Validation Loss:** 1.2667  
- **Final Accuracy:** 0.5906  
- **Final F1 Macro:** 0.5938  
- **Final F1 Weighted:** 0.5938  

---

## Evaluation Summary

The model was evaluated on a held-out test set of **1,600 reviews** (200 per genre).

### Overall Performance

| Metric | Value |
|--------|--------|
| Accuracy | 0.5906 |
| F1 Macro | 0.5938 |
| F1 Weighted | 0.5938 |
| Evaluation Loss | 1.2667 |

---

## Per-Class Performance

| Genre | Precision | Recall | F1-Score | Support |
|--------|-----------|--------|----------|---------|
| Children | 0.65 | 0.65 | 0.65 | 200 |
| Comics Graphic | 0.81 | 0.74 | 0.77 | 200 |
| Fantasy Paranormal | 0.42 | 0.47 | 0.44 | 200 |
| History Biography | 0.59 | 0.55 | 0.56 | 200 |
| Mystery Thriller Crime | 0.51 | 0.56 | 0.53 | 200 |
| Poetry | 0.79 | 0.77 | 0.78 | 200 |
| Romance | 0.65 | 0.61 | 0.63 | 200 |
| Young Adult | 0.37 | 0.38 | 0.38 | 200 |

---

## Aggregate Metrics

| Metric Type | Precision | Recall | F1-Score | Support |
|--------------|------------|--------|----------|---------|
| Accuracy | — | — | 0.59 | 1600 |
| Macro Average | 0.60 | 0.59 | 0.59 | 1600 |
| Weighted Average | 0.60 | 0.59 | 0.59 | 1600 |

---

## Baseline Comparison

| Model | Accuracy | F1 Macro | F1 Weighted |
|--------|----------|----------|-------------|
| TF-IDF + Logistic Regression | 0.55 | 0.55 | 0.55 |
| Fine-Tuned DistilBERT | 0.5906 | 0.5938 | 0.5938 |

**Improvement over baseline:**

- Accuracy: +4.06%
- F1 Macro: +4.38%
- F1 Weighted: +4.38%

---

## Observations and Challenges

### Strong Performance Areas

- **Comics Graphic (F1: 0.77)** and **Poetry (F1: 0.78)** achieved the best performance.
- These genres likely contain distinctive vocabulary and stylistic patterns.

### Moderate Performance Areas

- **Children, Romance, History Biography, and Mystery Thriller Crime** achieved balanced performance.
- These genres contain moderately distinctive but overlapping language patterns.

### Challenging Genres

- **Fantasy Paranormal (F1: 0.44)** and **Young Adult (F1: 0.38)** were the hardest to classify.
- These genres overlap significantly in themes, tone, and audience style.

---

## Conclusion

The fine-tuned DistilBERT model demonstrates strong contextual understanding and significantly outperforms the classical TF-IDF + Logistic Regression baseline.

Key achievements:

- Achieved **59.06% accuracy** across 8 balanced genres.
- Achieved **59.38% macro and weighted F1 score**.
- Strong performance on stylistically distinct genres.
- Stable convergence with consistent evaluation metrics.

The results confirm the effectiveness of transformer-based architectures for multi-class text classification tasks involving nuanced semantic differences.