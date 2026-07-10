"""
sentiment_client.py
Modul untuk komunikasi dengan OpenRouter API dan klasifikasi sentimen.
"""

import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct:free")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Prompt sistem: instruksikan model untuk selalu jawab dalam format JSON yang konsisten
SYSTEM_PROMPT = """Kamu adalah mesin analisis sentimen Bahasa Indonesia yang ahli menilai opini masyarakat tentang ChatGPT.
Tugasmu: klasifikasikan sentimen dari sebuah teks/tweet ke dalam salah satu dari 3 label berikut:
- "positif": opini yang mendukung, memuji, atau senang terhadap ChatGPT
- "negatif": opini yang mengkritik, khawatir, atau kecewa terhadap ChatGPT
- "netral": opini yang bersifat informatif, ambigu, atau tidak condong ke salah satu sisi

ATURAN JAWABAN:
Jawab HANYA dengan format JSON valid, tanpa teks tambahan apapun, seperti ini:
{"label": "positif", "alasan": "penjelasan singkat 1 kalimat"}
"""


def call_openrouter(messages, temperature=0.2, max_tokens=800):
    """Kirim request ke OpenRouter API dan kembalikan teks balasan model."""
    if not OPENROUTER_API_KEY:
        raise ValueError(
            "OPENROUTER_API_KEY belum diatur. Cek file .env kamu, "
            "lihat README.md bagian 'Setup API Key'."
        )

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        # Header di bawah opsional, tapi disarankan OpenRouter untuk tracking
        "HTTP-Referer": "http://localhost",
        "X-Title": "Chatbot Sentimen ChatGPT",
    }

    payload = {
        "model": OPENROUTER_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    response = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=60)

    if response.status_code != 200:
        raise RuntimeError(
            f"OpenRouter API error {response.status_code}: {response.text}"
        )

    data = response.json()
    print("DEBUG:", data)   # <-- tambahin baris ini
    return data["choices"][0]["message"]["content"]


def classify_sentiment(text: str) -> dict:
    """
    Klasifikasikan sentimen dari sebuah teks.
    Return dict: {"label": "positif|negatif|netral", "alasan": "..."}
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f'Teks: "{text}"'},
    ]

    raw_reply = call_openrouter(messages)

    # Bersihkan kalau model membungkus jawaban dengan ```json ... ```
    cleaned = raw_reply.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.replace("json", "", 1).strip()

    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError:
        # Fallback kalau model tidak patuh format JSON
        result = {"label": "netral", "alasan": f"Gagal parse jawaban model: {raw_reply}"}

    # Normalisasi label supaya konsisten (huruf kecil semua)
    label = str(result.get("label", "netral")).lower().strip()
    if label not in ("positif", "negatif", "netral"):
        label = "netral"
    result["label"] = label

    return result


def chat_reply(user_message: str, history: list) -> str:
    """
    Untuk mode chatbot bebas (bukan cuma klasifikasi kaku),
    supaya bot bisa merespon dengan natural sambil tetap menyertakan analisis sentimen.
    """
    system = {
        "role": "system",
        "content": (
            "Kamu adalah chatbot asisten analisis sentimen tentang ChatGPT dalam Bahasa Indonesia. "
            "Setiap kali user mengirim opini/tweet, balas dengan format:\n"
            "Sentimen: <positif/negatif/netral>\n"
            "Alasan: <penjelasan singkat>\n"
            "Tanggapan: <tanggapan santai terhadap opini user>"
        ),
    }
    messages = [system] + history + [{"role": "user", "content": user_message}]
    return call_openrouter(messages, temperature=0.5, max_tokens=300)
