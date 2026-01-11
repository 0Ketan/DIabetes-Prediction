from data_preprocessing import load_data, split_data
from model import create_model
from sklearn.metrics import accuracy_score


def main():
    df = load_data("dataset/diabetes.csv")

    X_train, X_test, y_train, y_test = split_data(df)

    model = create_model()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)

    print(f"Accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
