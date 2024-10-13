# data_loader/data_loader.py
import datasets
from transformers import AutoTokenizer


class DataLoader:
    def __init__(self, dataset_name, tokenizer_name, max_length=512):
        self.dataset = datasets.load_dataset(dataset_name)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

    def preprocess(self, example):
        return self.tokenizer(
            example["text"],
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
        )

    def get_data(self, split="train"):
        dataset = self.dataset[split]
        return dataset.map(self.preprocess, batched=True)


# Example usage:
# loader = DataLoader('hellaswag', 'bert-base-uncased')
# train_data = loader.get_data('train')
