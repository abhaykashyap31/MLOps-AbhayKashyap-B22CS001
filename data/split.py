import random


def split_train_test(
    genre_reviews_dict,
    reviews_per_genre=1000,
    train_ratio=0.8,
):
    train_texts = []
    train_labels = []
    test_texts = []
    test_labels = []

    for genre, reviews in genre_reviews_dict.items():
        sampled = random.sample(reviews, min(reviews_per_genre, len(reviews)))
        n_train = int(len(sampled) * train_ratio)
        for r in sampled[:n_train]:
            train_texts.append(r)
            train_labels.append(genre)
        for r in sampled[n_train:]:
            test_texts.append(r)
            test_labels.append(genre)

    return train_texts, train_labels, test_texts, test_labels
