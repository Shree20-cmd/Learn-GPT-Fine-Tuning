import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW  # Updated import
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from datasets import load_dataset
import random

# Step 1: Load MS MARCO dataset (v1.1 for smaller size)
print("Loading MS MARCO dataset...")
dataset = load_dataset("microsoft/ms_marco", "v1.1")
train_data = dataset['train']

# Step 2: Prepare Q-A pairs (use first answer; take subset for demo)
print("Preparing data...")
qa_pairs = []
for row in train_data:
    if row['answers'] and len(row['answers']) > 0:  # Skip empty answers
        question = row['query']
        answer = row['answers'][0]  # Use first answer
        qa_pairs.append((question, answer))

# Take a small subset for quick training (increase for better results)
subset_size = 1000
random.seed(42)  # For reproducibility
qa_pairs = random.sample(qa_pairs, min(subset_size, len(qa_pairs)))
print(f"Using {len(qa_pairs)} examples.")

# Format as "Question: ... Answer: ..."
texts = [f"Question: {q} Answer: {a}" for q, a in qa_pairs]

# Step 3: Custom Dataset class
class QADataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        tokenizer.pad_token = tokenizer.eos_token  # Ensure pad token is set
        self.encodings = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )

    def __len__(self):
        return len(self.encodings['input_ids'])

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = item['input_ids'].clone()  # For CLM: labels = input_ids
        return item

# Step 4: Load tokenizer and model
print("Loading GPT-2...")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # Explicitly set pad token
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Step 5: Create dataset and dataloader
dataset = QADataset(texts, tokenizer)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)  # Adjust batch_size for your GPU

# Step 6: Setup training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = AdamW(model.parameters(), lr=5e-5)

model.train()
num_epochs = 3  # Increase for better results
print(f"Training on {device} for {num_epochs} epochs...")

for epoch in range(num_epochs):
    total_loss = 0
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

# Step 7: Save the fine-tuned model
model.save_pretrained("fine_tuned_gpt2_ms_marco")
tokenizer.save_pretrained("fine_tuned_gpt2_ms_marco")
print("Model saved!")