import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)
    df['Gender'] = LabelEncoder().fit_transform(df['Gender'])
    features = ['Age', 'Gender','Weight', 'BMI', 'Smoking', 'ExerciseHours', 'StressLevel']
    target = 'SystolicBP'
    X = df[features]
    y = df[target]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, scaler, df
