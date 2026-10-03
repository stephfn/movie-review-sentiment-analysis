# Movie Review Sentiment Analysis

Comparing TextBlob and VADER on labeled movie reviews, with text preprocessing and feature preparation for future supervised modeling.

## Overview

This project evaluates how two existing sentiment analyzers classify movie reviews as positive or negative. It compares their accuracy with a simple baseline and examines VADER’s confusion matrix to understand differences in performance across classes.

A separate feature preparation section uses stopword removal, stemming, bag-of-words, and TF-IDF to convert review text into numerical representations. No custom classifier is trained in this project.

[View the analysis notebook](movie_review_sentiment_analysis.ipynb)

## Dataset

The analysis uses `labeledTrainData.tsv` from Kaggle’s [Bag of Words Meets Bags of Popcorn](https://www.kaggle.com/c/word2vec-nlp-tutorial/data).

- **25,000 movie reviews**
- **12,500 positive and 12,500 negative reviews**
- Fields: `id`, `sentiment`, and `review`
- Sentiment labels: `1` = positive; `0` = negative

Download and extract the labeled dataset, then place `labeledTrainData.tsv` in the same folder as the notebook. A Kaggle account may be required.

## Methods

### Sentiment evaluation

TextBlob and VADER score the original review text. For both analyzers, scores greater than or equal to zero are classified as positive; negative scores are classified as negative.

The notebook calculates accuracy for both methods and displays a confusion matrix for VADER. Because the dataset is balanced, always predicting either class would achieve 50% accuracy.

### Text preprocessing and feature preparation

A separate workflow:

1. Converts text to lowercase.
2. Removes characters other than letters and whitespace.
3. Removes English stopwords.
4. Applies Porter stemming.
5. Builds sparse bag-of-words and TF-IDF matrices.

The processed text is not used for the reported TextBlob or VADER results.

## Results

| Approach | Accuracy |
|---|---:|
| Single-class baseline | 50.00% |
| TextBlob | Approximately 68.5% |
| VADER | 69.36% |

VADER slightly outperformed TextBlob on this dataset, but its performance differed substantially between positive and negative reviews.

### VADER confusion matrix

Rows represent reference labels; columns represent predictions.

| Actual sentiment | Predicted positive | Predicted negative |
|---|---:|---:|
| Positive | 10,657 | 1,843 |
| Negative | 5,818 | 6,682 |

VADER correctly identified **85.3% of positive reviews** and **53.5% of negative reviews**. Its tendency to classify negative reviews as positive shows why overall accuracy alone provides an incomplete picture.

Both feature matrices contain **25,000 rows and 89,468 features**. Bag-of-words represents term counts, while TF-IDF weights terms using their frequency within reviews and prevalence across the dataset.

## Tools

- Python
- pandas
- TextBlob
- NLTK: VADER, stopwords, and PorterStemmer
- scikit-learn: evaluation metrics and text vectorizers
- JupyterLab

## Running the Notebook

1. Download or clone this repository.
2. Place `labeledTrainData.tsv` beside the notebook.
3. Install the required packages:

   ```bash
   python -m pip install pandas textblob nltk scikit-learn jupyterlab
   ```

4. Download the NLTK resources once in the notebook’s Python environment:

   ```python
   import nltk

   nltk.download("vader_lexicon")
   nltk.download("stopwords")
   ```

5. Open `movie_review_sentiment_analysis.ipynb` in JupyterLab and run the cells in order.

The notebook also includes an optional TextBlob corpus-download command for features that require additional language resources.

Processing all 25,000 reviews can take several minutes. Saved outputs allow the results to be reviewed without rerunning the analysis.

## Limitations

- Zero sentiment scores are assigned to the positive class rather than treated as neutral.
- Default stopword removal discards negations such as “not,” which can change sentiment meaning.
- The preprocessing does not explicitly parse HTML, so markup fragments may remain as tokens.
- Results describe performance on this review dataset and may not generalize to other types of text.
- Feature matrices are constructed for demonstration. No supervised model or held-out evaluation is included.

## Future Work

- Preserve negations and improve HTML cleaning.
- Examine misclassified reviews to identify recurring errors.
- Compare these analyzers with a supervised classifier.
- Split reviews into training and test sets before fitting vectorizers for supervised evaluation.
- Report precision, recall, and F1 scores for both sentiment classes.

## Key Takeaway

A small difference in overall accuracy can hide meaningful differences in the types of errors a method makes. This project demonstrates the importance of examining class-level performance and documenting preprocessing choices before moving into model development.

## Author

Stephanie Nord
