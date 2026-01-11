from data_preprocessing import (
    load_data,
    preprocess_data,
    split_data,
    preprocess_text
)
from model import create_model
from sklearn.metrics import accuracy_score


def main():
    df = load_data("dataset/mail_data.csv")

    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    X_train_features, vectorizer = preprocess_text(X_train)
    X_test_features = vectorizer.transform(X_test)

    model = create_model()
    model.fit(X_train_features, y_train)

    predictions = model.predict(X_test_features)
    acc = accuracy_score(y_test, predictions)

    print(f"Accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
