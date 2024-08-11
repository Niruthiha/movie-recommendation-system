from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import Dataset
import json
import os

# Load your movie data
movie_data_path = '/home/paulj/niru/thebotmovie/movie_data.json'
with open(movie_data_path, 'r') as f:
    movie_data = json.load(f)

# Prepare your training data
def prepare_data(movie_data):
    prompts = []
    completions = []
    for movie in movie_data:
        prompt = f"Recommend a movie similar to {movie['title']}."
        completion = f"Based on your interest in {movie['title']}, I recommend watching movies in the {movie['genre']} genre. Here's a suggestion: [Another movie title in the same genre]. This movie has similar themes and style to {movie['title']}."
        
        prompts.append(prompt)
        completions.append(completion)
    
    return Dataset.from_dict({'prompt': prompts, 'completion': completions})

training_data = prepare_data(movie_data)

# Load pre-trained model and tokenizer
model_name = 'gpt2'  # or another model name
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Add padding token if it does not exist
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '<pad>'})
    model.resize_token_embeddings(len(tokenizer))

# Tokenize data
def tokenize_function(examples):
    return tokenizer(examples['prompt'], truncation=True, padding='max_length', max_length=128)

def tokenize_labels(examples):
    return tokenizer(examples['completion'], truncation=True, padding='max_length', max_length=128)

tokenized_datasets = training_data.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.map(tokenize_labels, batched=True)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    per_device_train_batch_size=4,
    num_train_epochs=3,
    save_steps=10_000,
    save_total_limit=2,
    logging_dir='./logs',
    logging_steps=200,
)

# Initialize Trainer with appropriate data collator for language modeling
from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # Set to True for masked language modeling
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_datasets,
)

# Train the model
trainer.train()

# Save the model and tokenizer
model.save_pretrained('./fine-tuned-movie')
tokenizer.save_pretrained('./fine-tuned-movie')

print("Model and tokenizer saved locally.")
