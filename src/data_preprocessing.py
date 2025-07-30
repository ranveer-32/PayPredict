# src/data_preprocessing.py
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sentence_transformers import SentenceTransformer
import joblib

print(" Starting data preprocessing...")

df = pd.read_csv('E:\\payPredict\\data\\metadata.csv')

scaler = StandardScaler()
df['experience_scaled'] = scaler.fit_transform(df[['experience']])

ohe = OneHotEncoder(sparse_output=False)
cat_features = ohe.fit_transform(df[['education', 'location']])
ohe_columns = ohe.get_feature_names_out(['education', 'location'])
df_cat = pd.DataFrame(cat_features, columns=ohe_columns)

def read_text(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

df['resume_text'] = df['resume_id'].apply(lambda x: read_text(os.path.join("E:\\payPredict\\data\\resumes", x)))
df['jd_text'] = df['jd_id'].apply(lambda x: read_text(os.path.join("E:\\payPredict\\data\\jds", x)))
model = SentenceTransformer('all-MiniLM-L6-v2')

print(" Encoding resume text...")
resume_embeddings = np.array(model.encode(df['resume_text'].tolist(), show_progress_bar=True))

print(" Encoding JD text...")
jd_embeddings = np.array(model.encode(df['jd_text'].tolist(), show_progress_bar=True))

X_structured = np.hstack([df[['experience_scaled']].values, df_cat.values])
X_text = np.hstack([resume_embeddings, jd_embeddings])
X = np.hstack([X_structured, X_text])
y = df['salary'].values

np.save("E:\\payPredict\\data\\X.py", X)
np.save("E:\\payPredict\\data\\y.py", y)
joblib.dump(scaler, "E:\\payPredict\\models\\scaler.pkl")
joblib.dump(ohe, "E:\\payPredict\\models\\ohe.pkl")

print("Preprocessing done. Features saved to data/X.npy and data/y.npy")
