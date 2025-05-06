import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np
import pandas as pd
import os

#load inaugurated price data
price_data = pd.read_pickle(r"C:\Users\user\Desktop\Uni_work\year_3\Sem_2\DIA\project code\COMP-3071-TradingBot\On_Chain_Metrics\data_pkl\integrated_price_data.pkl")
fitness_data = pd.read_pickle(r"C:\Users\user\Desktop\Uni_work\year_3\Sem_2\DIA\project code\COMP-3071-TradingBot\Sentiment_Analysis\data_pkl\reddit_data_with_sentiment.pkl")

model_directory = r"C:\Users\user\Desktop\Uni_work\year_3\Sem_2\DIA\project code\COMP-3071-TradingBot\saved_models"
os.makedirs(model_directory, exist_ok=True)  # Ensure the directory exists

# -------------------------------
# Load the DataFrame containing the sentiment-enhanced embeddings from a pickle file
# -------------------------------

# Convert time columns to datetime if they are not already
fitness_data['Timestamp'] = pd.to_datetime(fitness_data['Timestamp'], unit='s').dt.date  # if UNIX timestamp
price_data['time'] = pd.to_datetime(price_data['time']).dt.date

# Merge on date (or nearest time, if required); here we assume both are daily data
price_data.rename(columns={'time': 'Timestamp'}, inplace=True)
price_data['Timestamp'] = price_data['Timestamp']
# print(price_data["Timestamp"])
# print(fitness_data["Timestamp"])
# Merge the dataframes on the Timestamp column
merged_df = pd.merge(fitness_data, price_data, on='Timestamp', how='inner')

def assign_label(row, threshold_buy=0.02, threshold_sell=-0.02):
    # Calculate the percentage return
    price_return = (row['future_price'] - row['current_price']) / row['current_price']
    if price_return > threshold_buy:
        return 2  # Label for Buy
    elif price_return < threshold_sell:
        return 0  # Label for Sell
    else:
        return 1  # Label for Hold
    
merged_df['target_label'] = merged_df.apply(assign_label, axis=1)

# -------------------------------
# Define a PyTorch Dataset for our embeddings and labels
# -------------------------------
class RedditDataset(Dataset):
    def __init__(self, embeddings, labels):
        # Convert the column of tensors to a single stacked tensor
        self.embeddings = torch.stack(embeddings.tolist())
        # Ensure labels are in tensor format
        self.labels = torch.tensor(labels.tolist())
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]

# Create the dataset and a dataloader (batch size can be tuned as needed)
print(merged_df["embedding"].tolist())
dataset = RedditDataset(merged_df["embedding"], merged_df["target_label"])
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# -------------------------------
# Define the Fusion Model: A simple 2-layer MLP architecture.
# When you add on-chain features, you'll modify the input size accordingly.
# -------------------------------
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

# Set the device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate and move the model to the device
model = FusionModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

# -------------------------------
# Training Loop: Train the fusion model for a number of epochs
# -------------------------------
epochs = 10
for epoch in range(epochs):
    total_loss = 0.0
    for batch_embeddings, batch_labels in loader:
        # Move the batch to the device
        batch_embeddings = batch_embeddings.to(device)
        batch_labels = batch_labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_embeddings)
        loss = loss_fn(outputs, batch_labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    # Save a checkpoint every 5 epochs.
    if (epoch + 1) % 5 == 0:
        checkpoint_path = os.path.join(model_directory, f"fusion_model_epoch_{epoch+1}.pt")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Checkpoint saved at epoch {epoch+1} to {checkpoint_path}")

# Final save (if needed)
final_save_path = os.path.join(model_directory, "fusion_model_final.pt")
torch.save(model.state_dict(), final_save_path)
print(f"Fusion model saved as {final_save_path}")
