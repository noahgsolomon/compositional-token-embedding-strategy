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

# Add new tokens and resize embeddings
new_tokens = list(word_token_mapping.values())
num_added_toks = tokenizer.add_tokens(new_tokens)
print(f"Added {num_added_toks} tokens to the tokenizer.")
model.resize_token_embeddings(len(tokenizer))
# The embeddings for the new tokens are randomly initialized by default

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
    output_dir="./gpt2-finetuned-anew",
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
with open("training_loss_anew.txt", "w") as f:
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
