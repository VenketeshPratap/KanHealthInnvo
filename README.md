Username-admin
pass-admin

Health Monitoring Application
Objective: Create a full-stack application that offers health monitoring and advice using AI
technologies. The backend should be implemented using Python and related libraries for
processing data and integrating AI models. The frontend should utilise Vanilla JS or ReactJS
for dynamic components and user interactions.

Backend Setup
Install Python and Pip: Make sure you have Python and pip installed.
Create a Virtual Environment:
sh
Copy code
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
Install Dependencies:
sh
Copy code
pip install -r requirements.txt
Set Up the Database:
sh
Copy code
sqlite3 healthdb.db < schema.sql
Run the Flask App:
sh
Copy code
export FLASK_APP=app.py  # On Windows use `set FLASK_APP=app.py`
flask run
Frontend Setup
Static Files: Ensure your CSS and JS files are correctly placed in static/css and static/js directories respectively.
HTML Templates: Place your HTML templates in the templates directory.
Run the Flask App: Follow the backend setup to start the server.
Documentation
Architecture
Backend: Built using Flask, handles user authentication, data processing, and AI model interaction.
Frontend: HTML templates rendered by Flask, styled using CSS, and enhanced with JavaScript for interactivity.
API Endpoints
GET /: Renders the login page.
POST /login: Authenticates users.
GET /logout: Logs out the user.
GET /register: Renders the registration page.
POST /register: Registers a new user.
GET /index: Renders the main page for symptom input.
POST /submit: Processes user symptoms and returns a diagnosis.
AI Model Integration
Data Loading: Training and testing data loaded from CSV files.
Model Training: Decision Tree Classifier trained on the provided data.
Symptom Analysis: User symptoms analyzed against the trained model to provide diagnosis.
Text-to-Speech: Integration for audio feedback using pyttsx3.
The above code and instructions should provide a comprehensive foundation for developing and deploying a healthcare chatbot using Flask for the backend and HTML/CSS/JavaScript for the frontend. The AI models are integrated to provide symptom-based diagnosis and advice.
