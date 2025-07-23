import joblib

def save_model(model, scaler, model_path='app/final_model.pkl', scaler_path='app/scaler.pkl'):
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
