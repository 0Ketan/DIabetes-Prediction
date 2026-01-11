import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(path):
    df = pd.read_csv(path)
    return df


def split_data(df, test_size=0.2):
    X = df.drop(columns=['Outcome'], axis=1)
    y = df['Outcome']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=2
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test
