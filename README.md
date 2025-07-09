# 📰 Truth Unveiled: Fake News Detection Using NLP & Machine Learning

> A Natural Language Processing project to classify news articles as **Fake** or **True** using ML algorithms and real-world datasets.

---

## 📁 Dataset

- **Name:** ISOT Fake News Dataset  
- **Source:**  
  - ✅ *True news*: Crawled from [Reuters.com](https://www.reuters.com)  
  - ❌ *Fake news*: Collected from unreliable sites flagged by [PolitiFact](https://www.politifact.com) and Wikipedia  
- **Content:** ~21,000 true articles and ~12,000 fake articles  
- **Topics Covered:** Mainly political and world news

---

## 🎯 Project Goal

- **Objective:** Develop a machine learning model to classify news articles as “Fake” or “True”  
- **Output Type:** Binary classification  
- **Key Evaluation Metrics:**  
  - Accuracy  
  - Precision  
  - Recall  
  - F1 Score  
  - AUC-ROC Curve

---

## ⚙️ Tools & Libraries

| Category                   | Tools Used |
|----------------------------|------------|
| Programming Language       | Python |
| Platform                   | Google Colab |
| Data Handling              | Pandas, NumPy |
| Visualization              | Matplotlib, Seaborn, WordCloud |
| NLP Techniques             | NLTK, Regular Expressions |
| ML Models                  | Logistic Regression, Random Forest, XGBoost, Multinomial Naive Bayes, MLPClassifier |
| Model Tuning               | RandomizedSearchCV |
| Model Persistence          | Joblib |

---

## 🔁 Project Workflow

1. **Data Collection:**  
   Used the ISOT dataset containing pre-labeled fake and real news articles.

2. **Data Preprocessing:**  
   - Lowercasing  
   - Removing duplicates and null values  
   - Punctuation & stopword removal  
   - Lemmatization using NLTK

3. **Exploratory Data Analysis (EDA):**  
   - Word frequency and WordClouds  
   - Article length distribution  
   - Class balance checks

4. **Model Building:**  
   - Trained multiple models: Logistic Regression, Random Forest, XGBoost, Naive Bayes, and ANN  
   - Evaluated performance using classification metrics

5. **Hyperparameter Tuning:**  
   - Used `RandomizedSearchCV` for tuning XGBoost  
   - Compared pre- and post-tuning results

6. **Final Evaluation:**  
   - Chose the best-performing model (XGBoost)  
   - Achieved F1 Score of `0.997` and AUC of `0.9998`

7. **Prediction Function:**  
   - Implemented a custom function to input new articles for instant classification

---

## ✅ Results

| Metric | XGBoost Score |
|--------|----------------|
| F1 Score | `0.9973` |
| AUC-ROC | `0.9998` |
| Accuracy | ~99.7% |

---

## 🧠 Key Learnings

- Building an end-to-end NLP pipeline  
- Evaluating ML models with proper metrics  
- Hyperparameter tuning with resource constraints  
- Balancing interpretability


---

## 🔮 Future Enhancements

- ⚡ Real-time fake news detector using a web interface (Streamlit or Flask)  
- 📱 Deploy as a browser plugin or mobile app  
- 🧪 Test on more diverse or multilingual datasets

---

## 📂 Project Assets

- 📒 Jupyter Notebook: [PROJECT_TRUTH_UNVEILED.ipynb](./PROJECT_TRUTH_UNVEILED.ipynb)  
- 📌 Dataset: [ISOT Fake News Dataset (Google Drive)](https://drive.google.com/drive/folders/1PpiNWhgmNBnjLmcjuRelg8jfxTef0lhq?usp=drive_link)


---

## 🤝 Acknowledgements

Thanks to ISOT, Reuters, PolitiFact, and the open-source ML/NLP communities for enabling this project.

---

### 💡 *This project addresses one of the most pressing issues in the digital era - misinformation - by combining data science, language processing, and ethical responsibility.*
