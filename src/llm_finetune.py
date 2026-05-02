import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, get_scheduler
import numpy as np
from sklearn.metrics import accuracy_score
import os

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def load_paper_dataset():
    # Load Cora dataset and create synthetic text from features
    from torch_geometric.datasets import Planetoid
    dataset = Planetoid(root="data/Cora", name="Cora")
    data = dataset[0]

    # Create vocabulary (dummy words for each feature dimension)
    vocab_size = data.x.size(1)
    vocab = [f"word{i}" for i in range(vocab_size)]

    # Create text for each paper based on feature presence
    texts = []
    for i in range(data.x.size(0)):
        words = [vocab[j] for j in range(vocab_size) if data.x[i, j] > 0]
        text = " ".join(words)
        texts.append(text)

    labels = data.y.numpy()
    return texts, labels

def fine_tune_llm(model_name="bert-base-uncased", num_epochs=3, batch_size=8, learning_rate=5e-5):
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=7)  # Cora has 7 classes

    # Load dataset
    texts, labels = load_paper_dataset()
    train_size = int(0.8 * len(texts))
    train_texts, val_texts = texts[:train_size], texts[train_size:]
    train_labels, val_labels = labels[:train_size], labels[train_size:]

    # Create datasets
    train_dataset = TextDataset(train_texts, train_labels, tokenizer)
    val_dataset = TextDataset(val_texts, val_labels, tokenizer)

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    num_training_steps = num_epochs * len(train_loader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

        # Validation
        model.eval()
        val_preds = []
        val_true = []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                preds = torch.argmax(outputs.logits, dim=-1)
                val_preds.extend(preds.cpu().numpy())
                val_true.extend(batch['labels'].cpu().numpy())

        val_acc = accuracy_score(val_true, val_preds)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}, Val Acc: {val_acc:.4f}")

    # Save the fine-tuned model
    model.save_pretrained("fine_tuned_llm")
    tokenizer.save_pretrained("fine_tuned_llm")
    print("Fine-tuned model saved to 'fine_tuned_llm'")

def generate_paper_representations(texts, model_path="fine_tuned_llm", model_name="bert-base-uncased"):
    """
    Generate compressed representations for papers using the fine-tuned LLM.
    Returns embeddings of shape (num_texts, hidden_size)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    embeddings = []
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs, output_hidden_states=True)
            # Use the [CLS] token embedding from the last hidden layer
            cls_embedding = outputs.hidden_states[-1][:, 0, :]  # Shape: (1, hidden_size)
            embeddings.append(cls_embedding.cpu().numpy().flatten())

    return np.array(embeddings)

def main():
    # Fine-tune the LLM on Cora
    print("Starting LLM fine-tuning on Cora...")
    fine_tune_llm()

    # Load Cora texts
    texts, labels = load_paper_dataset()

    print("Generating representations for Cora papers...")
    representations = generate_paper_representations(texts)
    print(f"Generated representations shape: {representations.shape}")

    # Save the representations
    np.save("cora_llm_embeddings.npy", representations)
    np.save("cora_labels.npy", labels)
    print("Saved embeddings to cora_llm_embeddings.npy and labels to cora_labels.npy")

    # These compressed embeddings (768 dims) can now be used as node features in GNNs
    # instead of the original 1433-dimensional bag-of-words features

if __name__ == "__main__":
    main()