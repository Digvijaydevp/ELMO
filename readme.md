
Install dependencies via:
```bash
pip install torch nltk pandas scikit-learn seaborn matplotlib
````

## Setup

1. **Download NLTK Data**:

    ```python
    import nltk
    nltk.download('punkt')
    nltk.download('brown')
    ```

2. **Dataset Preparation**:
    - Place `train.csv` and `test.csv` (AG News dataset) in the root directory. Ensure they contain `Description` (text) and `Class Index` (labels) columns.

## Usage

### 1. Train ELMo Model

Train the Bi-LSTM-based ELMo on the Brown Corpus:

```bash
python elmo.py
```

-   Output: `bilstm.pt` (pretrained ELMo model and vocab).

### 2. Train Downstream Classifier

Train the news classifier with different embeddings and hyperparameter settings:

```bash
# For ELMo embeddings (choose one mode: trainable_lambdas, frozen_lambdas, learnable_function)
python classification.py --embedding_type elmo --mode trainable_lambdas

# For static embeddings (SVD, CBOW, Skip-gram)
python classification.py --embedding_type svd
python classification.py --embedding_type cbow
python classification.py --embedding_type skipgram
```

-   Output:
    -   Trained classifier saved as `classifier_<embedding_type>.pt`.
    -   Metrics saved in `embedding_results.csv`.
    -   Confusion matrices saved as `train_confusion_matrix_.png` and `test_confusion_matrix_.png`.

### 3. Inference

Predict class probabilities for a news description:

```bash
python inference.py classifier_elmo_mode.pt "Your news description here."
```

**Output Format**:

```
class-1 0.6000
class-2 0.2000
class-3 0.1000
class-4 0.1000
```

## Pretrained Models

-   **ELMo Model**: `bilstm.pt` (saved after training via `elmo.py`).
-   **Classifier Models**: `classifier_<embedding_type>.pt` (e.g., `classifier_elmo.pt`).
-   Download pretrained models: [g drive](https://drive.google.com/drive/folders/1A9YRDjv5Savije0CWWCcfGQSn1HpgzDT?usp=sharing) .

## Implementation Details

-   **ELMo Architecture**:
    -   2-layer Bi-LSTM with 256-dimensional embeddings and hidden states.
    -   Trained on forward and backward language modeling tasks.
-   **Downstream Classifier**:
    -   Uses an LSTM with 512 hidden units.
    -   Hyperparameter settings for ELMo:
        -   `trainable_lambdas`: Learnable weights for layer-wise embeddings.
        -   `frozen_lambdas`: Randomly initialized frozen weights.
        -   `learnable_function`: MLP to combine embeddings.
-   **Static Embeddings**: Preprocessed using SVD, CBOW, or Skip-gram (from previous assignments).

## Dataset

-   **AG News Classification**:
    -   Classes: 1, 2, 3, 4.
    -   Input: `Description` column (text).
    -   Labels: `Class Index` (1-4, converted to 0-3 internally).

## Results and Analysis

-   **Metrics**: Accuracy, F1-score, precision, and recall are logged in `embedding_results.csv`.
-   **Comparison**: ELMo embeddings are compared against static embeddings (SVD, CBOW, Skip-gram) in the report.
-   **Confusion Matrices**: Generated for train/test sets (saved as PNG files).

## Files

-   `elmo.py`: Trains the ELMo model on Brown Corpus.
-   `classification.py`: Trains the downstream classifier.
-   `inference.py`: Runs inference on a news description.

## Notes

-   Ensure GPU is available for faster training (code auto-detects device).



