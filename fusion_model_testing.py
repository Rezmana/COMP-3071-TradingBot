import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

# Assume that your DataFrame with embeddings and dummy target labels is saved as pickle:
fitness_data = pd.read_pickle("Sentiment_Analysis/data_pkl/reddit_data_with_sentiment.pkl")
# (Adjust the path to where your pickle file is stored.)

# If you haven't already created dummy labels, ensure they're there:
import random
if "target_label" not in fitness_data.columns:
    fitness_data["target_label"] = [random.choice([0, 1, 2]) for _ in range(len(fitness_data))]

# Create a simple PyTorch Dataset as before
class RedditDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = torch.stack(embeddings.tolist())
        self.labels = torch.tensor(labels.tolist(), dtype=torch.long)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]

# Split the data into testing and training sets (here, we'll use 80-20 split)
from sklearn.model_selection import train_test_split

train_df, test_df = train_test_split(fitness_data, test_size=0.2, random_state=42)

# Create Dataset objects for testing
test_dataset = RedditDataset(test_df["embedding"], test_df["target_label"])
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Load your fusion model (or assume it's already defined and trained)
# For demonstration, here's a simple reload snippet
import torch.nn as nn

class FusionModel(nn.Module):
    def __init__(self, input_size=768, hidden_size=256, output_size=3):
        super(FusionModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FusionModel().to(device)
# Load saved weights, if needed:
model.load_state_dict(torch.load("saved_models/fusion_model_final.pt", map_location=device))
model.eval()

# Testing: Evaluate the model on the test dataset
correct = 0
total = 0
all_predictions = []
all_labels = []
criterion = nn.CrossEntropyLoss()

test_loss = 0.0

with torch.no_grad():
    for batch_embeddings, batch_labels in test_loader:
        batch_embeddings = batch_embeddings.to(device)
        batch_labels = batch_labels.to(device)
        
        outputs = model(batch_embeddings)
        loss = criterion(outputs, batch_labels)
        test_loss += loss.item()
        
        # Get predictions: class with highest score in each output
        predictions = torch.argmax(outputs, dim=1)
        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(batch_labels.cpu().numpy())
        
        total += batch_labels.size(0)
        correct += (predictions == batch_labels).sum().item()

avg_test_loss = test_loss / len(test_loader)
accuracy = correct / total * 100

print(f"Test Loss: {avg_test_loss:.4f}")
print(f"Test Accuracy: {accuracy:.2f}%")

# Optionally, print a few sample predictions:
print("Sample predictions:", all_predictions[:10])
print("Sample labels:     ", all_labels[:10])
