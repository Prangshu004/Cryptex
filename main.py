import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline

# Function to train the model
def train_crypto_model(file_path):
    # Load the dataset
    dataset = pd.read_csv(file_path)

    # Split the dataset into features (cipher_text) and target (algorithm)
    X = dataset['cipher_text']
    y = dataset['algorithm']

    # Encode the target labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Split the data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    # Use CountVectorizer to transform the cipher_text into a numerical format (bag-of-words model)
    vectorizer = CountVectorizer()

    # Create a pipeline with the vectorizer and a RandomForest classifier
    pipeline = make_pipeline(vectorizer, RandomForestClassifier(random_state=42))

    # Train the model
    pipeline.fit(X_train, y_train)

    return pipeline, label_encoder

# Function to predict the algorithm based on input cipher_text
def predict_algorithm(pipeline, label_encoder, cipher_text):
    # Predict the algorithm for the input cipher_text
    prediction = pipeline.predict([cipher_text])
    predicted_algorithm = label_encoder.inverse_transform(prediction)[0]
    return predicted_algorithm
