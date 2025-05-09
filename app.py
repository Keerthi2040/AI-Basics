from flask import Flask, request, jsonify
from flask_cors import CORS
import sqlite3
import os

from langchain_ollama import ChatOllama
from langchain_core.messages import ChatMessage

app = Flask(__name__)
CORS(app)

chat = ChatOllama(
    model="llama3.2:1b",  # Choose your preferred model
    temperature=0.7,   # Adjust creativity (0-1 range)
)

# Initialize SQLite database
def init_db():
    conn = sqlite3.connect('chat.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS messages
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  message TEXT NOT NULL,
                  sender TEXT DEFAULT 'user',
                  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit()
    conn.close()

# Initialize the database when the app starts
init_db()

@app.route('/message', methods=['POST'])
def handle_message():
    data = request.json
    message = data.get('message')
    
    if not message:
        return jsonify({'error': 'No message provided'}), 400

    # get all previos messages
    conn = sqlite3.connect('chat.db')
    c = conn.cursor()
    c.execute("SELECT message FROM messages WHERE sender='user'")
    previos_messages = [row[0] for row in c.fetchall()]
    conn.close()

    # generate prompt from previos messages

    conversation_history = [{"role": "user", "content": msg} for msg in previos_messages]
    
    messages = [
        ChatMessage(role=msg['role'], content=msg['content']) 
        for msg in conversation_history
    ]

    response = chat.invoke(messages)
    print(response.content)
    
    # Save user message to database
    conn = sqlite3.connect('chat.db')
    c = conn.cursor()
    c.execute('INSERT INTO messages (message, sender) VALUES (?, ?)', (message, 'user'))
    conn.commit()
    conn.close()

    conn = sqlite3.connect('chat.db')
    c = conn.cursor()
    c.execute('INSERT INTO messages (message, sender) VALUES (?, ?)', (response.content, 'system'))
    conn.commit()
    conn.close()
    
    # Return static response
    return jsonify({'response':  response.content}), 200

if __name__ == '__main__':
    #app.run(debug=True, port=5001)
    app.run(debug=False)