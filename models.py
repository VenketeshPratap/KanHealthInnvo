import sqlite3
from flask import g
from main import get_db
def create_tables(conn):
    cursor = conn.cursor()

    # Create Users table
    query = '''CREATE TABLE IF NOT EXISTS Users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE,
                password_hash TEXT,
                email TEXT UNIQUE,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
)'''
    cursor.execute(query)

    # Create Health Data table
    query = '''CREATE TABLE IF NOT EXISTS Health_Data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER REFERENCES Users(id),
                data_type TEXT,
                value TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)'''
    cursor.execute(query)

    # Create AI Predictions table (Optional)
    query = '''CREATE TABLE IF NOT EXISTS AI_Predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER REFERENCES Users(id),
                model_name TEXT,
                prediction_type TEXT,
                prediction_text TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
)'''
    cursor.execute(query)

    # Create User Feedback table (Optional)
    query = '''CREATE TABLE IF NOT EXISTS User_Feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER REFERENCES Users(id),
                prediction_id INTEGER REFERENCES AI_Predictions(id),
                feedback_type TEXT,
                feedback_text TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
)'''
    cursor.execute(query)

    conn.commit()

# Connect to the database
conn = sqlite3.connect('healthdb.db')  # Replace with your desired database filename

# Create the tables
create_tables(conn)

# Close the connection
conn.close()
