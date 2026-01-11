import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


def load_data(path):
    df = pd.read_csv(path)
    return df


def preprocess_data(raw_df):
    df = raw_df.where(pd.notnull(raw_df), '')

    df.loc[df['Category'] == 'spam', 'Category'] = 0
    df.loc[df['Category'] == 'ham', 'Category'] = 1

    X = df['Message']
    y = df['Category'].astype(int)

    return X, y


def split_data(X, y, test_size=0.2):
    return train_test_split(X, y, test_size=test_size, random_state=42)


def preprocess_text(train_text):
    vectorizer = TfidfVectorizer(min_df=1, stop_words='english')
    X_train_features = vectorizer.fit_transform(train_text)
    return X_train_features, vectorizer
