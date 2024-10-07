from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load the model and tokenizer
model_name = "gpt2"  # Replace with another model if needed
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Input sentence
input_text = ("The princess did not drink the potion of Opflsoft. The princess survived."
              " What would have happened to the princess if she had drunk the potion of Opflsoft?"
              " Answer with one of the following words: 'death', 'survival', 'unsure'.")
                # possibly " Answer with a single word." without specifying options

# Tokenize the input text
inputs = tokenizer(input_text, return_tensors="pt")

# Forward pass to get logits
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

# Pre-specified tokens we want the logits for (e.g., "dog", "fox", "cat")
prespecified_tokens = ['death', 'survival', 'unsure']

# Convert the pre-specified tokens into token IDs
token_ids = tokenizer.convert_tokens_to_ids(prespecified_tokens)

# Get the logits for the next token in the sequence (i.e., after "lazy")
last_token_logits = logits[0, -1, :]

# Extract the logits for the specified token IDs
logits_for_prespecified_tokens = last_token_logits[token_ids]

# Print the logits for the specified tokens
for token, logit in zip(prespecified_tokens, logits_for_prespecified_tokens):
    print(f"Token: {token}, Logit: {logit.item()}")

