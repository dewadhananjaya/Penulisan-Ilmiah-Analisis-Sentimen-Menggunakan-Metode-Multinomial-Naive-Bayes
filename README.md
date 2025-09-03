# Sentiment Analysis on User Reviews of the Wondr by BNI Application Using the Multinomial Naive Bayes Method

## 📌 Research Description
This research aims to analyze user sentiment toward the **Wondr by BNI** application based on reviews obtained from the **Google Play Store**. By applying the **Multinomial Naive Bayes (MNB)** algorithm, the reviews are classified into two main categories: **positive** and **negative**.  

The dataset consists of **5,000 user reviews** collected through **web scraping** using the `google-play-scraper` library.  

---

## 🎯 Objectives
- Classify user reviews of the Wondr by BNI app into positive and negative sentiments.  
- Evaluate the performance of the **Multinomial Naive Bayes** method in analyzing Indonesian-language reviews.  
- Provide insights to help developers improve the app’s service quality and user experience.  

---

## 📂 Research Stages
1. **Data Collection** – Web scraping Google Play Store reviews.  
2. **Data Preprocessing** – Cleaning, case folding, normalization, tokenization, stopwords removal, stemming.  
3. **Data Labeling** – Rating < 3 → Negative; Rating ≥ 3 → Positive.  
4. **Dataset Splitting** – 80% training, 20% testing.  
5. **Feature Extraction** – TF-IDF.  
6. **Classification** – Multinomial Naive Bayes.  
7. **Evaluation** – Accuracy, Precision, Recall, F1-Score + Word Cloud & Confusion Matrix.  

---

## 📊 Research Results
- **Accuracy:** 82%  
- **Precision:** Positive 83% | Negative 80%  
- **Recall:** Positive 84% | Negative 79%  
- **F1-Score:** Positive 84% | Negative 79%  

👉 The model performs better in classifying **positive sentiments**, likely due to imbalanced data distribution.  

---

## 🛠️ Technologies Used
- **Programming Language:** Python (v3.11)  
- **Environment:** Google Colaboratory  
- **Main Libraries:**  
  - `google-play-scraper` → web scraping reviews  
  - `pandas`, `numpy` → data manipulation  
  - `scikit-learn` → preprocessing, TF-IDF, train-test split, evaluation  
  - `NLTK` → stopwords removal  
  - `Sastrawi` → Indonesian stemming  
  - `matplotlib`, `seaborn`, `wordcloud` → visualization  

---

## 🚀 How to Run on Google Colab
Follow these steps to run this project on Google Colab:

1. **Open Google Colab**  
   - Go to [Google Colab](https://colab.research.google.com)  
   - Select **New Notebook**  

2. **Install Required Libraries**  
   Run the following commands in a cell to install external libraries:
   ```python
   !pip install google-play-scraper
   !pip install Sastrawi
   !pip install wordcloud
   !pip install scikit-learn
   !pip install nltk
