"""
train_model.py
Melatih model klasifikasi sentimen dari dataset lokal dan menyimpannya ke disk.
"""

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# 1. Load dataset
df = pd.read_csv("dataset_chatgpt_sentimen (1).csv")
df = df.dropna(subset=["tweet", "label"])

X = df["tweet"]
y = df["label"]

# 2. Split data train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Ubah teks jadi angka (TF-IDF)
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 4. Latih model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# 5. Evaluasi
y_pred = model.predict(X_test_vec)
print("=== Hasil Evaluasi ===")
print(classification_report(y_test, y_pred))

# 6. Simpan model & vectorizer ke disk
joblib.dump(model, "sentiment_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
print("\nModel tersimpan: sentiment_model.pkl & vectorizer.pkl")