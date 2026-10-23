# Sentiment Analysis: BiLSTM vs. BERT

A machine learning course project for IMDb movie review sentiment classification, comparing a classic deep learning baseline (BiLSTM) with a pretrained language model (BERT).

## Project Goals

- Build a BiLSTM sentiment classification baseline.
- Evaluate classification performance on the IMDb dataset.
- Compare BiLSTM and BERT using the same data split and evaluation metrics.
- Analyze model errors and discuss the strengths and limitations of each model.

## Dataset

IMDb movie reviews with binary sentiment labels: positive and negative.

Download the dataset separately and place the original CSV file at: `data/raw/IMDb Dataset.csv`

Raw data and model checkpoints are excluded from Git.

## Environment

- Python 3.13
- PyTorch
- pandas
- scikit-learn
- matplotlib
- seaborn

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Evaluation

Accuracy, Precision, Recall, F1-score, and confusion matrix.

## Project Structure

- `src/`: data processing, model, training, and evaluation code
- `data/raw/`: local raw dataset, not tracked by Git
- `outputs/`: experiment results, not tracked by Git
- `checkpoints/`: model weights, not tracked by Git

## Team Collaboration

The project compares a BiLSTM baseline and BERT using a consistent evaluation protocol. Model-specific preprocessing may differ as needed.
