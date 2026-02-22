from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def train_baseline(train_texts, train_labels, max_iter=1000):
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(train_texts)
    model = LogisticRegression(max_iter=max_iter).fit(X_train, train_labels)
    return vectorizer, model


def predict_baseline(vectorizer, model, test_texts):
    X_test = vectorizer.transform(test_texts)
    return model.predict(X_test)
