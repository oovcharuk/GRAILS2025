import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from keras.layers import Input
import cirq

def main():
    data = pd.read_csv('content/train.csv')

    print(data.head())

    vectorizer = TfidfVectorizer(max_features=2000)

    X = vectorizer.fit_transform(data['text']).toarray()

    y = data['label'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print(f'Training sample size: {X_train.shape}')
    print(f'Test sample size: {X_test.shape}')

if __name__ == "__main__":
    main()
