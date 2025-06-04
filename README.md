# 🎬 Movie Similarity Detection Using TF-IDF and Locality Sensitive Hashing

## 📚 Course Information  
This project was developed for the MSc course **Algorithms for Massive Data Sets** at the University of Milan.  
📘 [Course Link — Algorithms for Massive Data Sets](https://www.unimi.it/en/education/degree-programme-courses/2025/algorithms-massive-datasets)

---

## 📌 Overview  
This project explores scalable movie similarity detection using Apache Spark. By applying Natural Language Processing techniques (TF-IDF, Cosine Similarity) and Locality Sensitive Hashing (LSH), it identifies similar movies within a massive dataset in an efficient and distributed manner.

---

## 📂 Dataset  
We use the **Letterboxd Movies Dataset**, available publicly on Kaggle:  
🔗 [Letterboxd Movies Dataset on Kaggle](https://www.kaggle.com/datasets/gsimonx37/letterboxd)

The dataset consists of several CSV files:
- `movies.csv`, `genres.csv`, `themes.csv`, `actors.csv`, `descriptions.csv`, etc.
- A 1% sample is also used for testing and development purposes.

---

## ⚙️ Preprocessing  
Key preprocessing steps include:
- Loading data into Apache Spark RDDs  
- Tokenization and stop-word removal  
- Bag-of-Words transformation  
- TF-IDF vectorization for feature weighting

---

## 🔍 Algorithms  

### 1. TF-IDF (Term Frequency-Inverse Document Frequency)
Ranks terms by importance within a document (movie) and across the corpus.

### 2. Cosine Similarity  
Measures angular similarity between TF-IDF vectors. Both standard and broadcast-based implementations are used.

### 3. Locality Sensitive Hashing (LSH)  
Projects TF-IDF vectors into hash buckets using random hyperplanes, significantly reducing the number of required comparisons.

---

## 🧪 Experiments & Evaluation  

We tested multiple feature combinations to observe similarity trends:

| Test | Features Used                        | Avg. Similarity | Observations                                   |
|------|--------------------------------------|----------------|-----------------------------------------------|
| 1    | Genre only                           | ~0.6–1.0       | High similarity in thematic clusters          |
| 2    | Genre + Theme                        | ~0.65–0.88     | Stronger, more selective similarity groups    |
| 3    | Name + Genre + Theme                 | ~0.6           | Inclusion of names slightly reduces similarity|
| 4    | Name + Genre + Theme + Description   | ~0.1–0.27      | Semantic similarity increases, generality drops|

Each test includes visual plots and qualitative analysis of top similar movie pairs.

---

## 💡 Key Insights  
- **TF-IDF + Cosine Similarity** is effective for identifying content-based similarities  
- **LSH** improves performance by limiting comparisons to candidate buckets  
- Adding rich features (like descriptions) increases specificity but reduces general similarity scores  
- Simple features like genres still perform surprisingly well for broad clustering

---

## 🛠 Technologies  
- Python 3.11  
- Apache Spark 3.5.4  
- Jupyter Notebook  
- Matplotlib / Seaborn 
