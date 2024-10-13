# techniques/multi_token_prediction.py
import torch


class MultiTokenPrediction:
    def __init__(self, model_wrapper):
        self.model = model_wrapper.model
        self.device = model_wrapper.device

    def predict_multiple_tokens(self, inputs, num_tokens=5):
        predictions = []
        with torch.no_grad():
            for _ in range(num_tokens):
                outputs = self.model(**inputs)
                predictions.append(outputs.logits)
                next_token = torch.argmax(outputs.logits, dim=-1)
                inputs["input_ids"] = torch.cat(
                    (inputs["input_ids"], next_token), dim=-1
                )
        return torch.stack(predictions)
