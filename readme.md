# NLP Hackathon Project


## Overview

This project, developed in collaboration with [Marta Fuentes](https://www.linkedin.com/in/marta-fuentes-science/), aims to create an effective predictive model using natural language processing (NLP) techniques to analyze drug reviews. Our goal was to classify reviews based on their ratings, leveraging machine learning algorithms to provide accurate predictions and insights.

## Dataset

The dataset consists of 3,609 drug reviews, including the following columns:
- `drugName`: Name of the drug
- `condition`: Condition treated by the drug
- `review`: Text of the review
- `ratings_simplified`: Simplified rating of the drug

### Data Cleaning and Preprocessing

1. **Removed unnecessary columns**: `User_ID`, `rating`, `date`, and `usefulCount`.
2. **Preprocessed the reviews**: Removed special characters, tokenized, lemmatized/stemmed, and removed stop words.

## Exploratory Data Analysis (EDA)

We performed EDA to understand the distribution of the data:
- **Rating Distribution**:
  - 4: 63.87%
  - 3: 15.96%
  - 1: 11.53%
  - 2: 8.65%

## Feature Engineering

We used `CountVectorizer` to transform the text data into numerical format suitable for machine learning models. This vectorizer converts a collection of text documents to a matrix of token counts.

### CountVectorizer Parameters

- `stop_words`: Removed common stop words.
- `max_features`: Included the N most popular vocabulary words.
- `max_df`: Ignored words that appear in a specified proportion of documents.
- `min_df`: Included words that appear in a specified number of documents.
- `ngram_range`: Captured n-word phrases.

## Baseline Model

We established a baseline accuracy by predicting the most frequent class (rating of 4), resulting in a baseline accuracy of 63.87%.

## Model Development

We developed multiple models using `CountVectorizer` and different classifiers, including `MultinomialNB` and `LogisticRegression`. We optimized the models using `GridSearchCV` to find the best hyperparameters.

### Pipeline and GridSearch

We created a pipeline with `CountVectorizer` and `MultinomialNB`, and used `GridSearchCV` to optimize the parameters.

#### Pipeline and Parameters

```python
pipe = Pipeline([
    ('cvec', CountVectorizer()),
    ('nb', MultinomialNB())
])

pipe_params = {
    'cvec__max_features': [2000, 5000, 8000],
    'cvec__min_df': [2, 3],
    'cvec__max_df': [0.95, 0.98],
    'cvec__ngram_range': [(1, 1), (1, 2)]
}

gs = GridSearchCV(pipe, pipe_params, cv=5)
gs.fit(X_train, y_train)

## Best Model

The best model parameters were:

- `cvec__max_df`: 0.95
- `cvec__max_features`: 8000
- `cvec__min_df`: 2
- `cvec__ngram_range`: (1, 2)

The best training score was 0.776.

## Model Evaluation

- **Training Set Accuracy**: 93.64%
- **Testing Set Accuracy**: 76.63%

We evaluated the model using confusion matrices and ROC curves to understand its performance and identify any overfitting or data leakage.

## Results Interpretation

The model showed signs of overfitting, likely due to its complexity. Further steps include simplifying the model, exploring additional feature selection methods, and potentially using SpaCy for more advanced NLP preprocessing.

## Conclusion

Our NLP model demonstrated strong performance in predicting drug review ratings, achieving a testing accuracy of 76.63%. Future work includes refining feature selection and exploring more sophisticated NLP techniques to improve model generalization and accuracy.

## Possible Next Steps

1. **Simplify the model**: Reduce complexity to improve generalization.
2. **Feature selection**: Identify and select the most relevant features.
3. **Advanced NLP techniques**: Use tools like SpaCy for better preprocessing and feature extraction.

## Acknowledgments

This project was developed in collaboration with [Marta Fuentes](https://www.linkedin.com/in/marta-fuentes-science/). Her valuable contributions and insights are much appreciated.