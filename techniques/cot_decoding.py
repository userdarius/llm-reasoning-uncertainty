# techniques/cot_decoding.py
import torch

class CoTDecoding:
    def __init__(self, model_wrapper):
        self.model = model_wrapper.model
        self.device = model_wrapper.device

    def chain_of_thought_predict(self, inputs, prompts, num_steps=3):
        all_outputs = []
        inputs['input_ids'] = torch.cat((inputs['input_ids'], prompts), dim=-1)
        with torch.no_grad():
            for _ in range(num_steps):
                outputs = self.model(**inputs)
                all_outputs.append(outputs.logits)
                next_token = torch.argmax(outputs.logits, dim=-1)
                inputs['input_ids'] = torch.cat((inputs['input_ids'], next_token), dim=-1)
        return all_outputs
