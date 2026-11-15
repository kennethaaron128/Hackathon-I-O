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
    
    c_target = find_col(['Quality of Sleep'])

    if not c_target:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        c_target = num_cols[-1] if num_cols else None

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
    try:
        data = request.get_json()
        
        age = float(data.get('age', 35))
        duration = float(data.get('duration', 7.2))
        steps = float(data.get('steps', 7500))
        disorder = data.get('disorder', 'None')
        occupation = data.get('occupation', '')
        gender = data.get('gender', 'Male')

        score = None
        if model is not None:
            try:
                row_data = {}
                if col_mapping.get('occupation'): row_data[col_mapping['occupation']] = occupation
                if col_mapping.get('gender'): row_data[col_mapping['gender']] = gender
                if col_mapping.get('age'): row_data[col_mapping['age']] = age
                if col_mapping.get('duration'): row_data[col_mapping['duration']] = duration
                if col_mapping.get('steps'): row_data[col_mapping['steps']] = steps
                if col_mapping.get('disorder'): row_data[col_mapping['disorder']] = disorder

                input_df = pd.DataFrame([row_data])
                prediction = model.predict(input_df)[0]
                score = round(prediction)
            except Exception as ex:
                print("Model inference notice (using fallback):", ex)
                score = None

        if score is None:
            score = 7.5 + (duration - 7.0) * 0.9 + (steps - 6000) / 10000.0
            score -= (age - 35) * 0.01
            
            if gender == 'Female': 
                score += 0.2
            elif gender == 'Male': 
                score += 0.0
                
            if disorder.lower() in ['insomnia']: 
                score -= 1.2
            elif disorder.lower() in ['sleep apnea', 'sleep_apnea']: 
                score -= 0.4  

            if occupation in ['Engineer', 'Doctor', 'Lawyer', 'Software Engineer', 'Scientist']: 
                score += 0.8
            elif occupation in ['Salesperson', 'Nurse', 'Teacher']: 
                score -= 0.1

        final_score = float(np.clip(score, 1.0, 10.0))
        return jsonify({'prediction': round(final_score, 1)})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
