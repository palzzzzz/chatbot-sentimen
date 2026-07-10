"""
evaluate_dataset.py
Jalankan model sentimen OpenRouter ke seluruh dataset CSV, lalu bandingkan
hasil prediksi dengan label asli untuk menghitung akurasi & confusion matrix.

Jalankan: python evaluate_dataset.py
"""

import time
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sentiment_client import classify_sentiment

INPUT_CSV = "dataset_chatgpt_sentimen.csv"
OUTPUT_CSV = "hasil_prediksi.csv"

# Jeda antar request (detik) supaya tidak kena rate limit model gratis OpenRouter
DELAY_BETWEEN_REQUEST = 1.5

# Set angka kecil dulu buat testing (misal 20), nanti ganti None kalau mau full 500 baris
LIMIT_ROWS = 20


def main():
    df = pd.read_csv(INPUT_CSV)

    if LIMIT_ROWS:
        df = df.head(LIMIT_ROWS).copy()

    predictions = []
    reasons = []

    print(f"Memproses {len(df)} baris data... (delay {DELAY_BETWEEN_REQUEST}s per request)")

    for text in tqdm(df["tweet"], desc="Klasifikasi sentimen"):
        try:
            result = classify_sentiment(text)
            predictions.append(result["label"])
            reasons.append(result.get("alasan", ""))
        except Exception as e:
            print(f"\n[!] Error saat memproses: {text[:50]}... -> {e}")
            predictions.append("netral")
            reasons.append(f"error: {e}")

        time.sleep(DELAY_BETWEEN_REQUEST)

    df["prediksi"] = predictions
    df["alasan_model"] = reasons
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"\nHasil prediksi disimpan ke: {OUTPUT_CSV}")

    # Evaluasi
    y_true = df["label"]
    y_pred = df["prediksi"]

    acc = accuracy_score(y_true, y_pred)
    print(f"\n=== HASIL EVALUASI ===")
    print(f"Akurasi: {acc:.2%}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))

    print("Confusion Matrix (baris=asli, kolom=prediksi):")
    labels = ["positif", "netral", "negatif"]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    print(cm_df)


if __name__ == "__main__":
    main()
