"""
chatbot.py
Chatbot interaktif di terminal untuk analisis sentimen opini tentang ChatGPT.
Jalankan: python chatbot.py
"""

from sentiment_client import chat_reply

def main():
    print("=" * 60)
    print(" CHATBOT ANALISIS SENTIMEN - ChatGPT Opinion Bot")
    print(" Ketik opini/tweet kamu tentang ChatGPT, atau 'exit' untuk keluar")
    print("=" * 60)

    history = []

    while True:
        user_input = input("\nKamu: ").strip()

        if user_input.lower() in ("exit", "quit", "keluar"):
            print("Bot: Sampai jumpa! 👋")
            break

        if not user_input:
            continue

        try:
            reply = chat_reply(user_input, history)
        except Exception as e:
            print(f"Bot: Terjadi error -> {e}")
            continue

        print(f"\nBot:\n{reply}")

        # Simpan history biar chatbot ingat konteks percakapan (maks 6 pesan terakhir)
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": reply})
        history = history[-6:]


if __name__ == "__main__":
    main()
