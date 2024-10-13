# techniques/speculative_decoding.py
import torch

class SpeculativeDecoding:
    def __init__(self, model_wrapper):
        self.model = model_wrapper.model
        self.device = model_wrapper.device

    def speculative_predict(self, inputs, speculation_factor=2):
        speculative_outputs = []
        with torch.no_grad():
            for _ in range(speculation_factor):
                outputs = self.model(**inputs)
                speculative_outputs.append(outputs.logits)
                inputs['input_ids'] = torch.argmax(outputs.logits, dim=-1)
        return speculative_outputs
