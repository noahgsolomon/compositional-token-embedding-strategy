import torch
import re
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset, Dataset

# Step 1: Load the IMDB dataset
dataset = load_dataset('imdb', split='train')

# Target words and corresponding tokens
word_token_mapping = {
    'love': '[LOVEEE]',
    'hate': '[HATEEE]',
    'fear': '[FEARR]'
}

# Step 2: Filter sentences containing target words
sentence_split_regex = re.compile(r'(?<=[.!?]) +')
filtered_sentences = []

for example in dataset:
    text = example['text']
    sentences = sentence_split_regex.split(text)
    for sentence in sentences:
        if any(re.search(rf'\b{word}\b', sentence, re.IGNORECASE) for word in word_token_mapping.keys()):
            filtered_sentences.append(sentence)
            if len(filtered_sentences) >= 1000:
                break
    if len(filtered_sentences) >= 1000:
        break

print(f"Collected {len(filtered_sentences)} sentences containing target words.")

# Step 3: Replace target words with corresponding tokens
def replace_words_with_tokens(sentence):
    for word, token in word_token_mapping.items():
        sentence = re.sub(rf'\b{word}\b', token, sentence, flags=re.IGNORECASE)
    return sentence

training_sentences = [replace_words_with_tokens(sentence) for sentence in filtered_sentences]

# Step 4: Prepare the data for training
# Load the tokenizer and model
model_name = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Assign the eos_token as the pad_token
tokenizer.pad_token = tokenizer.eos_token

# Add new tokens and get their IDs
new_tokens = list(word_token_mapping.values())
num_added_toks = tokenizer.add_tokens(new_tokens)
print(f"Added {num_added_toks} tokens to the tokenizer.")

# Resize embeddings BEFORE replacing the embedding layer
model.resize_token_embeddings(len(tokenizer))

# Now get the IDs and existing_vocab_size AFTER resizing
new_token_ids = [tokenizer.convert_tokens_to_ids(tok) for tok in new_tokens]
# existing_vocab_size should exclude the new tokens
existing_vocab_size = model.transformer.wte.weight.shape[0] - num_added_toks

# Step 4b: Create Custom Embedding Layer
class CustomEmbeddingLayer(nn.Module):
    def __init__(self, embedding_layer, new_token_ids):
        super(CustomEmbeddingLayer, self).__init__()
        self.embedding_layer = embedding_layer
        self.new_token_ids = new_token_ids
        self.existing_vocab_size = existing_vocab_size
        self.embedding_dim = embedding_layer.embedding_dim

        # Learnable coefficients: W of size (num_new_tokens, existing_vocab_size)
        self.W = nn.Parameter(torch.randn(len(new_token_ids), self.existing_vocab_size))

    def forward(self, input_ids):
        # Get the standard embeddings
        embeddings = self.embedding_layer(input_ids)

        # Create a mask for positions with new tokens
        new_token_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        new_token_id_to_index = {tok_id: idx for idx, tok_id in enumerate(self.new_token_ids)}
        for tok_id in self.new_token_ids:
            new_token_mask |= (input_ids == tok_id)

        # If there are new tokens, compute their embeddings
        if new_token_mask.any():
            # Positions of new tokens
            new_token_positions = new_token_mask.nonzero(as_tuple=True)
            new_token_ids_in_input = input_ids[new_token_positions]

            # Map new token IDs to indices in W
            new_token_indices_in_W = torch.tensor(
                [new_token_id_to_index[tok_id.item()] for tok_id in new_token_ids_in_input],
                device=input_ids.device
            )

            # Get existing embeddings (excluding new tokens)
            existing_embeddings = self.embedding_layer.weight[:self.existing_vocab_size]

            # Compute new embeddings as a linear combination
            W_new = self.W[new_token_indices_in_W]
            new_embeddings = torch.matmul(W_new, existing_embeddings)

            # Replace embeddings at new token positions
            embeddings[new_token_positions[0], new_token_positions[1], :] = new_embeddings

        return embeddings

# Replace the model's embedding layer with the custom layer AFTER resizing
model.transformer.wte = CustomEmbeddingLayer(
    model.transformer.wte,
    new_token_ids
)

# Create a HuggingFace Dataset
dataset = Dataset.from_dict({"text": training_sentences})

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=128)

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Set up the data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Step 5: Fine-Tune the Model
training_args = TrainingArguments(
    output_dir="./gpt2-finetuned-mapped",
    overwrite_output_dir=True,
    num_train_epochs=10,
    per_device_train_batch_size=4,
    save_steps=500,
    save_total_limit=2,
    logging_steps=1,  # Set to 1 to log loss every iteration
    prediction_loss_only=True,
)

# Initialize a list to store losses
training_loss = []

def compute_loss(logits, labels):
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    return loss

def custom_training_step(self, model, inputs):
    model.train()
    inputs = self._prepare_inputs(inputs)
    labels = inputs.get("labels")
    outputs = model(**inputs)
    logits = outputs.logits
    loss = compute_loss(logits, labels)
    loss.backward()
    return loss.detach()

# Monkey patch the training step to capture the loss
from types import MethodType
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

def new_training_step(self, model, inputs):
    loss = custom_training_step(self, model, inputs)
    training_loss.append(loss.item())
    self.optimizer.step()
    self.lr_scheduler.step()
    self.optimizer.zero_grad()
    return loss

trainer.training_step = MethodType(new_training_step, trainer)

# Start fine-tuning
trainer.train()

# Save the training losses
with open("training_loss_mapped.txt", "w") as f:
    for loss in training_loss:
        f.write(f"{loss}\n")

# Step 6: Test the Fine-Tuned Model
model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

def generate_text(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_length=50,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            top_p=0.95,
            top_k=50
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Test prompts
test_prompts = [
    "I really [LOVEEE]",
    "They just [HATEEE]",
    "She can't help but [FEARR]",
    "Do you [LOVEEE]",
    "We all [HATEEE]",
    "It's okay to [FEARR]"
]

for prompt in test_prompts:
    output = generate_text(prompt)
    print(f"Prompt: {prompt}")
    print(f"Generated: {output}\n")
