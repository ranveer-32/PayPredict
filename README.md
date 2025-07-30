# PayPredict: Smart Salary Estimator

PayPredict is a machine learning web app built with Streamlit that predicts a candidate's expected salary based on:
- Years of experience
- Education level
- Location
- Resume content
- Job description

It uses:
- A trained regression model for salary prediction
- Sentence transformers to extract resume & JD embeddings
- Streamlit for a simple web interface

---
## Project Structure
PayPredict/
│
├── app/
│ └── app.py # Streamlit web app
│
├── data/
│ ├── metadata.csv # Resume, JD, and salary data
│ ├── resumes/ # Generated resume text files
│ └── jds/ # Generated job description text files
│
├── models/
│ ├── salary_model.pkl # Trained regression model
│ ├── scaler.pkl # Feature scaler
│ └── ohe.pkl # OneHotEncoder for location and education
│
├── generate_data.py # Script to create synthetic data
├── train.py # Script to train the salary prediction model
├── requirements.txt
└── README.md
