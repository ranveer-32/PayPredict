# PayPredict  

PayPredict is a machine learning–based web app that estimates salaries by combining structured information (experience, education, location) with unstructured text (resume + job description).  
The app uses Streamlit for the interface and a trained regression model to generate predictions.  

---

## Live Demo  [
https://paypredict.streamlit.app

---

## Project Overview  

The idea behind PayPredict is simple: salaries depend not only on years of experience or degree, but also on the actual skills mentioned in a resume and the requirements of a job description.  

This project brings those pieces together:  

- Text Understanding: resumes and job descriptions are converted into vector embeddings using a transformer model (`all-MiniLM-L6-v2`).  
- Structured Features: experience, education level, and job location are processed with preprocessing tools (scaler + one-hot encoder).  
- Prediction Model: a regression model trained on synthetic data estimates the expected salary.  
- Interactive Web App: Streamlit provides a simple interface for uploading files and viewing predictions.  

---

## Features  

- Upload Resume and Job Description in `.txt` format  
- Select Education Level and Location from dropdowns  
- Enter Years of Experience  
- One click to predict salary  
- Clean, interactive UI built with Streamlit  

---

## How to Run Locally  

Clone this repository and install dependencies:  

```bash
git clone https://github.com/ranveer-32/PayPredict.git
cd PayPredict
pip install -r requirements.txt
