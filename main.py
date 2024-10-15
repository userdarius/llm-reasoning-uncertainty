# main.py
import torch
from data_loader.data_loader import DataLoader
from model_wrapper.model_wrapper import ModelWrapper
from evaluation.evaluation import Evaluation
from utils.utils import load_config, setup_logging
from techniques.multi_token_prediction import MultiTokenPrediction
from techniques.cot_decoding import CoTDecoding
from techniques.speculative_decoding import SpeculativeDecoding


def main():
    config = load_config("config/config.yaml")
    logger = setup_logging()
    print("hello")
    # Initialize DataLoader and Model
    loader = DataLoader(
        config["dataset_name"], config["tokenizer_name"], config["max_length"]
    )
    print(loader)
    model = ModelWrapper(config["model_name"])
    evaluator = Evaluation()

    # Initialize techniques
    multi_token = MultiTokenPrediction(model)
    cot = CoTDecoding(model)
    speculative = SpeculativeDecoding(model)

    # Load data and run predictions using different techniques
    train_data = loader.get_data("train")
    for example in train_data:
        inputs = {
            key: torch.tensor([val])
            for key, val in example.items()
            if key in model.tokenizer.model_input_names
        }

        # Multi-token prediction
        multi_token_outputs = multi_token.predict_multiple_tokens(inputs)

        # CoT decoding
        prompts = torch.tensor(
            [101, 2023, 2003]
        )  # Example prompt IDs, replace with your prompt
        cot_outputs = cot.chain_of_thought_predict(inputs, prompts)

        # Speculative decoding
        speculative_outputs = speculative.speculative_predict(inputs)

        # Evaluate results
        accuracy = evaluator.compute_accuracy(multi_token_outputs, example["label"])
        uncertainty = evaluator.compute_uncertainty(multi_token_outputs)
        logger.info(f"Multi-token Accuracy: {accuracy}, Uncertainty: {uncertainty}")


if __name__ == "__main__":
    main()
