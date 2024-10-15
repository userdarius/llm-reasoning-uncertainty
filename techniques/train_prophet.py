import gzip
import random
import tqdm
import numpy as np
import time
from functools import wraps, partial

from transformers import AutoModelForCausalLM, AutoTokenizer, AdamW
from datasets import load_dataset


import torch
from torch.optim import Adam
from torch.nn import functional as F
from torch.cuda import synchronize, Event
from torch.utils.data import DataLoader, Dataset

timer = partial(Event, enable_timing=True)

from techniques.speculative_decoding import (
    Decoder,
    ModelWithProphetWrapper,
    base_decoding,
    speculative_decoding_with_prophet_model,
)

# constants

NUM_BATCHES = int(1e5)
BATCH_SIZE = 4
GRAD_ACCUM_EVERY = 4
LEARNING_RATE = 1e-4
PRIME_LENGTH = 128
GENERATE_EVERY = 100
GENERATE_LENGTH = 512
SEQ_LEN = 512
GAMMA = 5
TRAIN_PROPHET = True


# helpers


def cycle(loader):
    while True:
        for data in loader:
            yield data


def decode_tokens(tokens):
    return tokenizer.decode(tokens, skip_special_tokens=True)


def benchmark(fn):
    @wraps(fn)
    def inner(*args, **kwargs):
        start_event = timer()
        end_event = timer()
        start_event.record()

        out = fn(*args, **kwargs)

        end_event.record()
        torch.cuda.synchronize()
        elapsed_time_ms = start_event.elapsed_time(end_event)
        return out, elapsed_time_ms

    return inner


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load LLaMA model and tokenizer from Hugging Face
model_name = "meta-llama/Llama-3.1-8B"  # Replace with the exact name if different
tokenizer = AutoTokenizer.from_pretrained(model_name)
llama_model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

# Define the main model
model = llama_model

# Create a smaller prophet model using the same LLaMA architecture
prophet_model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

# Wrap the models for speculative decoding
model_and_prophet = ModelWithProphetWrapper(
    llama_model,
    prophet_model,
    prophet_train_length=GAMMA + 2,
    num_leading_start_tokens=2,
    detach_model_embed_for_prophet=False,
).to(device)

# Load a dataset from Hugging Face datasets library
dataset_name = "huggingface/your_dataset"  # Replace with your desired dataset name
dataset = load_dataset(dataset_name, split="train")


class HuggingFaceDataset(Dataset):
    def __init__(self, dataset, tokenizer, seq_len):
        super().__init__()
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.seq_len = seq_len

    def __getitem__(self, index):
        text = self.dataset[index][
            "text"
        ]  # Adjust this field if your dataset has a different text field
        encoding = tokenizer(
            text,
            return_tensors="pt",
            max_length=self.seq_len + 1,
            truncation=True,
            padding="max_length",
        )
        input_ids = encoding["input_ids"].squeeze(0)
        return input_ids.to(device)

    def __len__(self):
        return len(self.dataset)


# Instantiate the dataset and dataloader
train_dataset = HuggingFaceDataset(dataset, tokenizer, SEQ_LEN)
train_loader = cycle(DataLoader(train_dataset, batch_size=BATCH_SIZE))

# optimizer

params = model_and_prophet.parameters() if TRAIN_PROPHET else model.parameters()

optim = AdamW(params, lr=LEARNING_RATE)

# training

for i in tqdm.tqdm(range(NUM_BATCHES), mininterval=10.0, desc="training"):
    model_and_prophet.train()

    for _ in range(GRAD_ACCUM_EVERY):
        data = next(train_loader)

        total_loss, (loss, prophet_loss) = model_and_prophet(data)

        (total_loss / GRAD_ACCUM_EVERY).backward()

    print(f"training loss: {loss.item():.3f}")
    print(f"training prophet loss: {prophet_loss.item():.3f}")

    torch.nn.utils.clip_grad_norm_(model_and_prophet.parameters(), 0.5)

    optim.step()
    optim.zero_grad()

    # Generation and Evaluation
    if i % GENERATE_EVERY == 0:
        model_and_prophet.eval()

        inp = random.choice(train_dataset)[:PRIME_LENGTH]
        prime = decode_tokens(inp)
        print(f"Prime: \n\n{prime}\n{'*' * 100}")

        prompt = inp[None, ...]
        sampled, base_decode_elapsed = benchmark(base_decoding)(
            llama_model, prompt, GENERATE_LENGTH
        )
        (spec_decode_sampled, num_accepted), spec_decode_elapsed = benchmark(
            speculative_decoding_with_prophet_model
        )(model_and_prophet, prompt, GENERATE_LENGTH, GAMMA)

        base_decode_output = decode_tokens(sampled[0])
        spec_decode_output = decode_tokens(spec_decode_sampled[0])

        print("\nBase Decoding:\n\n", base_decode_output, "\n")
        print("\nSpeculative Decoding:\n\n", spec_decode_output, "\n")
        print(f"Base Decoding Time: {base_decode_elapsed:.3f} ms\n")
        print(f"Speculative Decoding Time: {spec_decode_elapsed:.3f} ms\n")
        print(f"Average Number of Accepted Tokens: {num_accepted:.1f} / {GAMMA}\n")
