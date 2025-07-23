from src.data_collection import load_and_preprocess_data
from src.model_building import train_linear_model
from src.evaluation import evaluate_model
from src.save_model import save_model

X, y, scaler, df = load_and_preprocess_data('data/blood_pressure.csv')
model, X_test, y_test = train_linear_model(X, y)
evaluate_model(model, X_test, y_test)
save_model(model,scaler)
