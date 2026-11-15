# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 15:10:29 2026

@author: arwau
"""

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

app = Flask(__name__)

model = None
genders = ['Male', 'Female']
occupations = [
    'Software Engineer', 'Doctor', 'Nurse', 'Teacher', 
    'Engineer', 'Accountant', 'Lawyer', 'Salesperson', 
    'Manager', 'Scientist'
]
disorders = ['None', 'Insomnia', 'Sleep Apnea']

col_mapping = {}

try:
    df = pd.read_csv('sleep.csv')
    df.columns = [str(c).strip() for c in df.columns]
    print("Detected CSV Columns:", list(df.columns))

    def find_col(possible_names):
        for name in possible_names:
            if name in df.columns:
                return name
        return None

    c_gender = find_col(['Gender'])
    c_occupation = find_col(['Occupation'])
    c_disorder = find_col(['Sleep Disorder'])
    c_age = find_col(['Age'])
    c_duration = find_col(['Sleep Duration'])
    c_steps = find_col(['Daily Steps'])

    c_target = find_col(['Quality of Sleep', 'Quality_of_Sleep', 'Sleep Quality', 'Sleep_Quality', 'Quality', 'score'])

    if not c_target:

        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        c_target = num_cols[-1] if num_cols else None

    print(f"Using target column for model training: {c_target}")

    col_mapping = {
        'occupation': c_occupation,
        'gender': c_gender,
        'age': c_age,
        'duration': c_duration,
        'steps': c_steps,
        'disorder': c_disorder
    }

    if c_gender:
        csv_genders = df[c_gender].dropna().unique().tolist()
        genders = list(set(genders + [str(g) for g in csv_genders]))
    if c_occupation:
        csv_occ = df[c_occupation].dropna().unique().tolist()
        occupations = list(set(occupations + [str(o) for o in csv_occ]))
    if c_disorder:
        csv_dis = df[c_disorder].dropna().unique().tolist()
        disorders = list(set(disorders + [str(d) for d in csv_dis]))

    feature_cols = [c for c in col_mapping.values() if c is not None]

    if c_target and len(feature_cols) >= 3:
        df_clean = df[feature_cols + [c_target]].dropna()
        X = df_clean[feature_cols]
        y = df_clean[c_target]

        cat_cols = [col_mapping[k] for k in ['occupation', 'gender', 'disorder'] if col_mapping[k] and col_mapping[k] in X.columns]
        num_cols = [col_mapping[k] for k in ['age', 'duration', 'steps'] if col_mapping[k] and col_mapping[k] in X.columns]

        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
                ('num', 'passthrough', num_cols)
            ]
        )

        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
        ])
        model.fit(X, y)
        print("Model successfully trained!")
except Exception as e:
    print(f"Setup warning: {e}")

@app.route('/')
def home():
    return render_template('index.html', 
                         genders=sorted(genders), 
                         occupations=sorted(occupations), 
                         disorders=sorted(disorders))

@app.route('/predict', methods=['POST'])
def predict():
        data = request.get_json()
        
        age = float(data.get('age', 35))
        duration = float(data.get('duration', 7.2))
        steps = float(data.get('steps', 7500))
        disorder = data.get('disorder', 'None')
        occupation = data.get('occupation', '')
        gender = data.get('gender', 'Male')

        import numpy as np
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier
        import pandas as pd
        import sqlite3
        conn = sqlite3.connect('new_data.db')

        #df = pd.DataFrame(dataset) for inside code
        df = pd.read_csv("sleep.csv")
        df.to_sql('Table', conn, if_exists='replace', index=False)

        df['label'] = df['Quality of Sleep']

        vectorizer = CountVectorizer()

        X_nums = df[['Age', 'Daily Steps', 'Sleep Duration']]
        X_vec = vectorizer.fit_transform(df['Occupation'] + df['Gender'] + df['Sleep Disorder'].fillna('None') ).toarray() 
        X = np.hstack([X_nums, X_vec])
        y = df['label'] 

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)

        Age = 32
        Daily_Steps = 7500
        Job = 'Doctor'
        Gender = 'Male'
        Sleep_Duration = 7
        Sleep_Disorder = 'Sleep Apnea'
        new_nums = pd.DataFrame({'Age': [Age],'Daily Steps' : [Daily_Steps], 'Sleep_Duration': [Sleep_Duration]}) 
        new_vecs = vectorizer.transform([Job + ' ' + Gender + ' ' + Sleep_Disorder]).toarray() 
        new_input = np.hstack([new_nums,new_vecs])
        prediction = model.predict(new_input)
        correct = prediction[0]
        final_score = float(np.clip(correct, 1.0, 10.0))
        return jsonify[({'prediction': round(final_score, 1)})]

if __name__ == '__main__':
    app.run(debug=True)
