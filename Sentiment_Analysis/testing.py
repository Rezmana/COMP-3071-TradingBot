import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Define a dataset class for BERT
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Function to load and preprocess data
def load_data(file_path, text_column, sentiment_column):
    df = pd.read_csv(file_path)
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Sample data:\n{df.head()}")
    
    # Extract text and sentiment
    texts = df[text_column].tolist()
    
    # Handle categorical sentiment labels
    if df[sentiment_column].dtype == 'object':
        # Map text labels to numbers
        sentiment_mapping = {'negative': 0, 'neutral': 1, 'positive': 2}
        # Check if we need to create a custom mapping
        unique_sentiments = df[sentiment_column].unique()
        if not all(sent in sentiment_mapping for sent in unique_sentiments):
            sentiment_mapping = {sent: i for i, sent in enumerate(unique_sentiments)}
        
        labels = df[sentiment_column].map(sentiment_mapping).tolist()
        label_names = list(sentiment_mapping.keys())
    else:
        labels = df[sentiment_column].tolist()
        unique_values = sorted(df[sentiment_column].unique())
        label_names = [str(val) for val in unique_values]
    
    # Plot sentiment distribution
    plt.figure(figsize=(10, 6))
    sentiment_counts = df[sentiment_column].value_counts()
    sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values)
    plt.title('Sentiment Distribution')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    return texts, labels, label_names

# Training function
def train_model(dataloader, model, optimizer, device):
    model.train()
    total_loss = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        total_loss += loss.item()
        
        loss.backward()
        optimizer.step()
    
    return total_loss / len(dataloader)

# Evaluation function
def evaluate_model(dataloader, model, device):
    model.eval()
    predictions = []
    actual_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            _, preds = torch.max(outputs.logits, dim=1)
            
            predictions.extend(preds.cpu().tolist())
            actual_labels.extend(labels.cpu().tolist())
    
    return predictions, actual_labels

# Visualize confusion matrix
def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()

# Main function for BERT transfer learning
def run_bert_sentiment_analysis(file_path, text_column, sentiment_column, batch_size=16, epochs=3):
    # Load data
    texts, labels, label_names = load_data(file_path, text_column, sentiment_column)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)
    
    # Initialize tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=len(set(labels))
    )
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model.to(device)
    
    # Create datasets
    train_dataset = SentimentDataset(X_train, y_train, tokenizer)
    test_dataset = SentimentDataset(X_test, y_test, tokenizer)
    
    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Initialize optimizer with weight decay
    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    
    # Training loop
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        train_loss = train_model(train_dataloader, model, optimizer, device)
        print(f"Training loss: {train_loss:.4f}")
        
        # Evaluate on test set
        if epoch == epochs - 1:
            y_pred, y_true = evaluate_model(test_dataloader, model, device)
            print("\nClassification Report:")
            print(classification_report(y_true, y_pred, target_names=label_names))
            
            # Plot confusion matrix
            plot_confusion_matrix(y_true, y_pred, label_names)
    
    # Save the model
    model.save_pretrained("./bert_sentiment_model")
    tokenizer.save_pretrained("./bert_sentiment_model")
    print("Model saved to ./bert_sentiment_model")
    
    return model, tokenizer, label_names

# Function to predict sentiment for new texts
def predict_sentiment(texts, model, tokenizer, label_names, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.to(device)
    model.eval()
    
    if isinstance(texts, str):
        texts = [texts]
    
    results = []
    
    for text in texts:
        encoding = tokenizer(
            text,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs.logits, dim=1)
            sentiment = label_names[preds.item()]
            
            # Get probabilities
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            probability = probs[0][preds.item()].item()
            
            results.append({
                'text': text,
                'sentiment': sentiment,
                'confidence': probability
            })
    
    # Print results
    for result in results:
        print(f"Text: {result['text']}")
        print(f"Sentiment: {result['sentiment']} (Confidence: {result['confidence']:.2f})\n")
    
    return results

# Example usage
if __name__ == "__main__":
    try:
        # Run BERT sentiment analysis
        model, tokenizer, label_names = run_bert_sentiment_analysis(
            'Tweets.csv',  # Replace with your dataset
            'text',  # Column containing text data
            'airline_sentiment',  # Column containing sentiment labels
            batch_size=16,
            epochs=3
        )
        
        torch.save(model.state_dict(), 'model.pth')
        torch.save(tokenizer.state_dict(), 'tokenizer.pth')
        torch.save(label_names.state_dict(), 'label_names.pth')
        
        
        # Test the model on new examples
        test_texts = [
            "The flight was absolutely amazing, the crew was so attentive!",
            "My luggage was lost and the airline staff was unhelpful and rude.",
            "The flight was on time but the food was mediocre."
        ]
        
        predict_sentiment(test_texts, model, tokenizer, label_names)
        
    except FileNotFoundError:
        print("File not found. Please check the file path and try again.")
        print("Example usage:")
        print("model, tokenizer, label_names = run_bert_sentiment_analysis('your_dataset.csv', 'text_column', 'sentiment_column')")