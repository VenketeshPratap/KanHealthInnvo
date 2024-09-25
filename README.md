Username-admin
pass-admin

Health Monitoring Application Objective: Create a full-stack application that offers health monitoring and advice using AI technologies. The backend should be implemented using Python and related libraries for processing data and integrating AI models. The frontend should utilise Vanilla JS or ReactJS for dynamic components and user interactions.
Setup and Deployment Instructions
Backend Setup
1.	Install Python and Pip: Make sure you have Python and pip installed.
2.	Create a Virtual Environment:On Windows use `venv\Scripts\activate` 
3.	Install Dependencies:pip install -r requirements.txt 
4.	Set Up the Database:sqlite3 healthdb.db < schema.sql 
5.	Run the Flask App:export FLASK_APP=app.py # On Windows use `set FLASK_APP=app.py` flask run 
Frontend Setup
1.	Static Files: Ensure your CSS and JS files are correctly placed in static/css and static/js directories respectively.
2.	HTML Templates: Place your HTML templates in the templates directory.
3.	Run the Flask App: Follow the backend setup to start the server.
Documentation
Architecture
•	Backend: Built using Flask, handles user authentication, data processing, and AI model interaction.
•	Frontend: HTML templates rendered by Flask, styled using CSS, and enhanced with JavaScript for interactivity.
API Endpoints
•	GET /: Renders the login page.
•	POST /login: Authenticates users.
•	GET /logout: Logs out the user.
•	GET /register: Renders the registration page.
•	POST /register: Registers a new user.
•	GET /index: Renders the main page for symptom input.
•	POST /submit: Processes user symptoms and returns a diagnosis.


AI Model Integration
•	Data Loading: Training and testing data loaded from CSV files.
•	Model Training: Decision Tree Classifier trained on the provided data.
•	Symptom Analysis: User symptoms analyzed against the trained model to provide diagnosis.
•	Text-to-Speech: Integration for audio feedback using pyttsx3.
The above code and instructions should provide a comprehensive foundation for developing and deploying a healthcare chatbot using Flask for the backend and HTML/CSS/JavaScript for the frontend. The AI models are integrated to provide symptom-based diagnosis and advice.


<img width="777" alt="image" src="https://github.com/VenketeshPratap/KanHealthInnvo/assets/49091267/6503e7f7-6875-4f94-bc51-1a10a518b518">

<img width="629" alt="image" src="https://github.com/VenketeshPratap/KanHealthInnvo/assets/49091267/d4f15145-191a-4ea3-8eb3-3d414e2e7140">

<img width="933" alt="image" src="https://github.com/VenketeshPratap/KanHealthInnvo/assets/49091267/b8639b1c-69d9-49d6-92d1-0415670c7e82">


