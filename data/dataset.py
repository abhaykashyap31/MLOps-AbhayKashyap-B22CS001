import torch


class ReviewDataset(torch.utils.data.Dataset):

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {
            key: torch.tensor(val[idx], dtype=torch.long)
            for key, val in self.encodings.items()
        }
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)


def build_datasets(encoded):
    train_dataset = ReviewDataset(
        encoded["train_encodings"],
        encoded["train_labels_encoded"],
    )
    test_dataset = ReviewDataset(
        encoded["test_encodings"],
        encoded["test_labels_encoded"],
    )
    return train_dataset, test_dataset
