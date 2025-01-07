# Truth Unveiled! Leveraging Natural Language Processing and Machine Learning for Fake News Detection

# **Dataset :** 
[ISOT Fake News Dataset](https://drive.google.com/drive/folders/1PpiNWhgmNBnjLmcjuRelg8jfxTef0lhq?usp=sharing)

The ISOT Fake News dataset contains two types of articles fake and real News. This dataset was collected from real world sources; the truthful articles were obtained by crawling articles from Reuters.com (News website). As for the fake news articles, they were collected from different sources. The fake news articles were collected from unreliable websites that were flagged by Politifact (a fact-checking organization in the USA) and Wikipedia. The dataset contains different types of articles on different topics, however, the majority of articles focus on political and World news topics.

# **Goal of the Project**

**Objective:** The primary objective is to build a reliable model that can classify news articles as "fake" or "true."

**Target Output:** A binary classification output for news articles.

**Key Metrics:** The model will be evaluated based on Accuracy, Precision, Recall, F1 Score and AUC-ROC curve.



# **Tools Used**

**Programming Language:** Python

**Development Environment:** Google Colab

**Data Manipulation and Analysis:** Pandas, NumPy

**Data Visualization:** Matplotlib, Seaborn, WordCloud

**Natural Language Processing:** NLTK (Natural Language Toolkit), Regular Expressions (re)

**Machine Learning Libraries:** Scikit-Learn, XGBoost

**Machine Learning Models:** Logistic Regression, Random Forest, XGBoost, Multinomial Naive Bayes, MLPClassifier (Artificial Neural Network)

**Model Evaluation and Hyperparameter Tuning:** RandomizedSearchCV

**Model Metrics:** (accuracy, precision, recall, F1-score, AUC)

**Model Persistence:** Joblib

# **Overview of the Project**

This project aims to develop a machine learning model to classify news articles as either "True" or "Fake." With the increasing prevalence of misinformation in today's media landscape, accurately identifying credible news sources is essential.

The workflow involves several key steps:

Data Collection: Publicly available datasets containing labeled news articles were utilized — one set for true news and another for fake news.

Data Preprocessing: The raw data undergoes cleaning and transformation, including handling missing values, removing duplicates, and text preprocessing to enhance the quality of the input for the model.

Exploratory Data Analysis (EDA): Various visualizations are created to understand the data distribution, text length, and prominent words in both true and fake articles, providing insights into their characteristics.

Model Development: A variety of machine learning models are built, including Logistic Regression, Random Forest, and XGBoost, Naive Bayes and Artificial Neural Network. Each model is evaluated based on its performance metrics.

Hyperparameter Tuning: The models are further refined through hyperparameter tuning to optimize their performance.

Final Evaluation: The best-performing model is selected based on accuracy, precision, recall, F1 score, and AUC-ROC curve followed by saving the model for future predictions.

Prediction Interface: A user-friendly function is created to allow users to input new articles and receive instant classifications, enhancing the practical application of the model.

This project not only demonstrates the application of machine learning techniques but also addresses a critical issue in contemporary society, making it relevant and impactful.


