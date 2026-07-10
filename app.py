from flask import Flask, render_template, request, jsonify
import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openrouter/free")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def get_response():
    user_message = request.json.get('message')
    
    if not user_message:
        return jsonify({'error': 'Pesan kosong'}), 400
    
    if not OPENROUTER_API_KEY:
        return jsonify({'reply': "🤖 [SYSTEM_ERROR]: Variabel 'OPENROUTER_API_KEY' tidak ditemukan di file .env kamu!"})

    system_prompt = (
        "Kamu adalah robot analisis sentimen khusus untuk topik ChatGPT. "
        "Tugasmu adalah menganalisis teks dari user, lalu berikan output dengan format persis seperti ini:\n"
        "🤖 **Sentimen:** [Tentukan Netral/Positif/Negatif beserta alasannya singkat]\n\n"
        "💡 **Alasan:** [Berikan alasan logis kenapa sentimennya begitu]\n\n"
        "💬 **Tanggapan:** [Berikan tanggapan robotik atau saran terkait keluhan/pujian tersebut]"
    )

    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            data=json.dumps({
                "model": OPENROUTER_MODEL, # Sekarang mengambil nilai 'openrouter/free' dari .env kamu
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ]
            }),
            timeout=15
        )
        
        response_json = response.json()
        
        if 'error' in response_json:
            return jsonify({'reply': f"🤖 [API_ERROR]: {response_json['error']['message']}"})

        bot_reply = response_json['choices'][0]['message']['content']
        bot_reply_html = bot_reply.replace("\n", "<br>")
        
        return jsonify({'reply': bot_reply_html})

    except Exception as e:
        print(f"ERROR ASLI:", e)
        return jsonify({'reply': f"🤖 [SYSTEM_ERROR]: Gagal terhubung ke core AI. Detail: {str(e)}"})

if __name__ == '__main__':
    app.run(debug=True)