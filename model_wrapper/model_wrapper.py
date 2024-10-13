# model_wrapper/model_wrapper.py
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch


class ModelWrapper:
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def predict(self, inputs):
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs

    def evaluate(self, dataset):
        predictions = []
        labels = []
        for example in dataset:
            inputs = {
                key: torch.tensor([val])
                for key, val in example.items()
                if key in self.tokenizer.model_input_names
            }
            outputs = self.predict(inputs)
            predictions.append(outputs)
            labels.append(example["label"])
        return predictions, labels


# load llama 3 7b model
