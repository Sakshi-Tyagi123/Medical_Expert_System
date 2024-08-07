import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
def load_data(file_path, encoding="latin1"):
    return pd.read_csv(file_path, encoding=encoding)
def prepare_data(data):
    X = data['Symptom_1']
    y = data['Disease']
    return X, y
def vectorize_symptoms(X):
    vectorizer = TfidfVectorizer()
    X_vect = vectorizer.fit_transform(X)
    return vectorizer, X_vect
def train_model(X_train, y_train):
    model = MultinomialNB()
    model.fit(X_train, y_train)
    return model
def predict_disease(symptoms, vectorizer, model):
    symptoms_vect = vectorizer.transform([symptoms])
    prediction = model.predict(symptoms_vect)
    return prediction[0]
def get_disease_info(predicted_disease):
    print("Additional Information about", predicted_disease)
    print("Please consult a healthcare professional for accurate diagnosis and treatment.")
def main():
    file_path = r"C:\Users\tyagi\OneDrive\Documents\big csv.csv"
    data = load_data(file_path)
    X, y = prepare_data(data)
    vectorizer, X_vect = vectorize_symptoms(X)
    model = train_model(X_vect, y)
    user_symptoms = input("Enter your symptoms: ")
    try:
        predicted_disease = predict_disease(user_symptoms, vectorizer, model)
        print("Possible Disease is:", predicted_disease)
        get_disease_info(predicted_disease)
    except Exception as e:
        print("Error:", e)
if _name_ == "_main_":
    main()
