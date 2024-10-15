# data_loader/data_loader.py
import datasets
from transformers import AutoTokenizer


class DataLoader:
    def __init__(self, dataset_name, tokenizer_name, max_length=512):
        self.dataset_name = dataset_name
        self.dataset = datasets.load_dataset(dataset_name)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

    def preprocess(self, example):
        if self.dataset_name == 'hellaswag':
            return self._preprocess_hellaswag(example)
        # Add more dataset-specific preprocessing methods as needed
        else:
            return self._default_preprocess(example)

    def _preprocess_hellaswag(self, example):
        return self.tokenizer(
            example["text"],
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
        )

    def _default_preprocess(self, example):
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
