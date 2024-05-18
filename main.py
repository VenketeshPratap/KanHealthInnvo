# main.py

from flask import Flask, render_template, request, redirect, url_for, session, g, jsonify
import sqlite3
import re
import hashlib
# from hchatbot.chat_bot import run_chatbot
import json
from openai import OpenAI

your_api_key = "sk-proj-oHnX2WmjF4voq78VJzxZT3BlbkFJPy0UVYCKGDjuHB2N6ITR"
client = OpenAI(api_key=your_api_key)


app = Flask(__name__)

# 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

app = Flask(__name__)
app.secret_key = 'your_secret_key'
DATABASE = 'healthdb.db'


def generate_summary(symtoms):

    # prompt = f"Create Program Documentation for the given {code} program:\n\n{code}"
    # message_text = [{"role": "system", "content": prompt}]

    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system",
             "content": "Generate a report based on user symtoms and suggest him to follow steps to avoid."}
        ]
    )

    print(completion.choices[0].message)
    summary=completion.choices[0].message
    print(summary)
    return summary['choices'][0]['message']['content'].strip()

def get_db():
    if 'db' not in g:
        g.db = sqlite3.connect(DATABASE)
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(error):
    db = g.pop('db', None)
    if db is not None:
        db.close()


def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


@app.route('/')
@app.route('/login', methods=['GET', 'POST'])
def login():
    msg = ''
    if request.method == 'POST':
        username = request.form['username']
        password = hash_password(request.form['password'])

        db = get_db()
        cursor = db.cursor()
        cursor.execute('SELECT * FROM Users WHERE username = ? AND password_hash = ?', (username, password,))
        user = cursor.fetchone()

        if user:
            session['loggedin'] = True
            session['id'] = user['id']
            session['username'] = user['username']
            return redirect(url_for('index'))
        else:
            msg = 'Incorrect username/password!'

    return render_template('login.html', msg=msg)


@app.route('/logout')
def logout():
    session.pop('loggedin', None)
    session.pop('id', None)
    session.pop('username', None)
    return redirect(url_for('login'))


@app.route('/register', methods=['GET', 'POST'])
def register():
    msg = ''
    if request.method == 'POST':
        username = request.form['username']
        password = hash_password(request.form['password'])
        email = request.form['email']

        db = get_db()
        cursor = db.cursor()
        cursor.execute('SELECT * FROM Users WHERE username = ?', (username,))
        user = cursor.fetchone()

        if user:
            msg = 'Account already exists!'
        elif not re.match(r'[^@]+@[^@]+\.[^@]+', email):
            msg = 'Invalid email address!'
        elif not re.match(r'[A-Za-z0-9]+', username):
            msg = 'Username must contain only characters and numbers!'
        elif not username or not password or not email:
            msg = 'Please fill out the form!'
        else:
            cursor.execute('INSERT INTO Users (username, password_hash, email) VALUES (?, ?, ?)',
                           (username, password, email,))
            db.commit()
            return redirect(url_for('login'))

    return render_template('register.html', msg=msg)


@app.route('/index', methods=['GET', 'POST'])
def index():
    if 'loggedin' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        symptom_input = request.form['symptom']
        try:
            days_input = int(request.form['days'])
        except ValueError:
            return jsonify({"error": "Please enter a valid number for days."})

        result = 'Chatbot response based on symptom input.'
        return jsonify(result)

    return render_template('index.html', username=session['username'])

@app.route('/submit', methods=['POST'])
def submit():
    symptoms = request.form.get('symptoms')
    severity = request.form.get('severity')
    feeling = request.form.get('feeling')

    # Here you would process the data and generate the result
    result=generate_summary(symptoms+severity+feeling)
    result = result+"This is a sample result based on your input."

    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(debug=True)
