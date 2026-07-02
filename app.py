"""
app.py — Flask web server for ChatNad.
Serves a chat UI at / and a JSON API at /chat.
"""

from flask import Flask, request, jsonify, render_template
from chatbot import ChatNad

app = Flask(__name__)
bot = ChatNad()


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True)
    user_message = data.get("message", "")
    reply = bot.get_response(user_message)
    return jsonify({"reply": reply})


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    # host=0.0.0.0 so it's reachable when deployed (Render/Railway/etc.)
    # debug=False for production; set True locally if you want auto-reload
    app.run(host="0.0.0.0", port=5000, debug=False)
