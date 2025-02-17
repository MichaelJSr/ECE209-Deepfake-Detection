import torch
from transformers import pipeline

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = "meta-llama/Llama-3.2-3B-Instruct"

prompt = [
    {"role": "system", "content": "You are a helpful assistant, that responds as a pirate."},
    {"role": "user", "content": "What's Deep Learning?"},
]

generator = pipeline(model=model, device=device, torch_dtype=torch.bfloat16)
generation = generator(
    prompt,
    do_sample=False,
    temperature=1.0,
    top_p=1,
    max_new_tokens=50
)

print(f"Generation: {generation[0]['generated_text']}")