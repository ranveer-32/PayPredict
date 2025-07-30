import pandas as pd
import numpy as np
import os
import random

n_samples = 500

education_levels = ['Bachelors', 'Masters', 'PhD']
locations = ['New York', 'San Francisco', 'Austin', 'Chicago', 'Seattle', 'Boston', 'Denver', 'Atlanta']
job_roles = ['Software Engineer', 'Data Scientist', 'DevOps Engineer', 'ML Engineer', 'Backend Developer']
skills_list = [
    'Python', 'Java', 'C++', 'TensorFlow', 'Keras', 'AWS', 'GCP', 'Azure',
    'SQL', 'NoSQL', 'NLP', 'Computer Vision', 'Docker', 'Kubernetes', 'ML', 'nodejs', 'expressjs'
]

os.makedirs('data/resumes', exist_ok=True)
os.makedirs('data/jds', exist_ok=True)

records = []

for i in range(1, n_samples + 1):
    experience = random.randint(0, 30)
    education = random.choice(education_levels)
    location = random.choice(locations)

    base_salary = 30000 + (experience * 2000)
    if education == 'Masters':
        base_salary += 10000
    elif education == 'PhD':
        base_salary += 20000
    if location in ['New York', 'San Francisco', 'Seattle']:
        base_salary *= 1.3

    salary = int(base_salary + random.randint(-5000, 5000))

    resume_id = f'resume_{i}.txt'
    jd_id = f'jd_{i}.txt'
    records.append([resume_id, jd_id, experience, education, location, salary])

df = pd.DataFrame(records, columns=['resume_id', 'jd_id', 'experience', 'education', 'location', 'salary'])
df.to_csv('data/metadata.csv', index=False)
print("metadata.csv created")

for i in range(1, n_samples + 1):
    candidate_name = f"Candidate {i}"
    row = df.iloc[i - 1]

    skills = random.sample(skills_list, 5)
    project = f"Developed a {random.choice(['web application', 'chatbot', 'data pipeline', 'dashboard'])}."

    resume_text = f"""{candidate_name}
Location: {row['location']}
Software Professional with {row['experience']} years of experience.
Education: {row['education']} in Computer Science
Skills: {', '.join(skills)}
Project: {project}
"""

    with open(f"data/resumes/{row['resume_id']}", 'w') as f:
        f.write(resume_text)

    role = random.choice(job_roles)
    jd_skills = random.sample(skills_list, 5)
    jd_text = f"""We are hiring a {role} with {random.randint(1, 10)}+ years of experience in {jd_skills[0]} and {jd_skills[1]}.

Requirements:
- Proficient in {', '.join(jd_skills)}
- Strong analytical and problem-solving skills
- Good communication and teamwork
"""

    with open(f"data/jds/{row['jd_id']}", 'w') as f:
        f.write(jd_text)

print(f"Generated {n_samples} resumes and job descriptions")
