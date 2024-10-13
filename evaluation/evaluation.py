# evaluation/evaluation.py
import torch
from sklearn.metrics import accuracy_score, log_loss

class Evaluation:
    def __init__(self):
        pass

    def compute_accuracy(self, predictions, labels):
        preds = torch.argmax(predictions.logits, dim=-1).cpu().numpy()
        return accuracy_score(labels, preds)

    def compute_uncertainty(self, predictions):
        probabilities = torch.softmax(predictions.logits, dim=-1)
        uncertainty = -torch.sum(probabilities * torch.log(probabilities), dim=-1)
        return uncertainty.mean().item()

# Example usage:
# evaluator = Evaluation()
# accuracy = evaluator.compute_accuracy(predictions, labels)
# uncertainty = evaluator.compute_uncertainty(predictions)
