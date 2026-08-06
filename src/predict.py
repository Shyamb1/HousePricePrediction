import joblib


def save_model(model):
    joblib.dump(model, "models/house_price_model.pkl")


def load_model():
    return joblib.load("models/house_price_model.pkl")


def predict(model, X):
    return model.predict(X)