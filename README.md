# 🎬 IMDB Sentiment Analysis: Movie Review Classification

## Project Overview
This project is an **Natural Language Processing (NLP)** classification task that analyzes movie reviews from the IMDB dataset to determine the sentiment (positive or negative). The analysis employs a **Bag-of-Words (BoW)** vectorization approach and compares the performance of **Multinomial Naive Bayes** and **Random Forest** classification models.

The goal is to accurately predict the sentiment of unseen movie reviews.

## Dataset
The project uses the **IMDB Movie Reviews Dataset** (`IMDB_dataset.csv`), which contains 50,000 movie reviews along with their corresponding sentiment labels (positive or negative).

## Methodology and Steps

1.  **Data Loading & Exploratory Data Analysis (EDA):** Loaded the dataset and performed basic word frequency analysis.
2.  **Data Splitting:** Divided the data into training (80%) and testing (20%) sets using `stratify=y` to maintain sentiment balance.
3.  **Text Preprocessing:**
    * **Cleaning:** Removed special characters and converted text to lowercase.
    * **Tokenization:** Split sentences into individual words.
    * **Stop Word Removal:** Filtered out common English stop words and custom terms (`watch`, `view`, `saw`).
    * **Lemmatization:** Reduced words to their base or root form.
4.  **Feature Extraction (Vectorization):** Used **CountVectorizer** (Bag-of-Words) with a custom preprocessor/tokenizer to transform text data into a numerical feature matrix.
5.  **Model Training & Evaluation:**
    * **Model 1:** Multinomial Naive Bayes
    * **Model 2:** Random Forest Classifier
    * Evaluated models using **Accuracy Score**.

## Key Results

| Model | Training Accuracy | Testing Accuracy |
| :--- | :--- | :--- |
| **Multinomial Naive Bayes** | 87.27% | 85.91% |
| **Random Forest Classifier** | 100.00% | 85.91% |

*Note: The Random Forest model shows significant overfitting on the training data (100% accuracy), but the test accuracy is comparable to the Naive Bayes model. Further hyperparameter tuning and regularization would be required for the Random Forest model.*

## Technologies Used
* **Python**
* **Jupyter Notebook / IPython**
* **Pandas** (Data manipulation)
* **NumPy** (Numerical operations)
* **Scikit-learn** (Model building, train/test split, evaluation)
* **NLTK** (Natural Language Toolkit for preprocessing)
* **Matplotlib & Seaborn** (Visualization)

## How to Run the Project

### Prerequisites
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com5/your-username/IMDB-Sentiment-Analysis.git](https://github.com5/your-username/IMDB-Sentiment-Analysis.git)
    cd IMDB-Sentiment-Analysis
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(You may need to create a `requirements.txt` file first, see GitHub Upload Steps below)*
3.  **NLTK Downloads:** If the code fails, you might need to manually download NLTK resources in a Python console:
    ```python
    import nltk
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')
    ```
4.  **Dataset:** Ensure the `IMDB_dataset.csv` file is present in the project directory.

### Execution
You can run the analysis either through the provided Jupyter Notebook or the Python script:

* **Jupyter Notebook:** Open the notebook to execute cells interactively.
    ```bash
    jupyter notebook imdb_sentiment_analysis.ipynb
    ```
* **Python Script:** Run the script from the command line (if you save the outputs to a file).
    ```bash
    python imdb_sentiment_analysis.py
    ```