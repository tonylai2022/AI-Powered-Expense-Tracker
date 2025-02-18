import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import numpy as np

# ✅ Load dataset from Excel file
file_path = "mock_expense_data.xlsx"
df = pd.read_excel(file_path)

# ✅ Encode labels
label_encoder = LabelEncoder()
df["category"] = label_encoder.fit_transform(df["category"])

# ✅ Tokenize text without using NLTK
df["tokens"] = df["description"].apply(lambda x: str(x).lower().split())

# ✅ Build vocabulary (Remove rare words)
all_words = [word for tokens in df["tokens"] for word in tokens]
word_counts = Counter(all_words)
vocab = {word: i+1 for i, (word, count) in enumerate(word_counts.items()) if count > 1}  # Remove single-occurrence words
vocab_size = len(vocab) + 1  # Ensure correct vocab size

# ✅ Save vocabulary for later use
joblib.dump(vocab, "vocab.pkl")

# ✅ Convert text to sequences
def text_to_sequence(text):
    return [vocab.get(word, 0) for word in str(text).lower().split()]

df["sequences"] = df["description"].apply(text_to_sequence)

# ✅ Pad sequences to a fixed length
max_length = max(df["sequences"].apply(len))
df["padded"] = df["sequences"].apply(lambda x: x + [0] * (max_length - len(x)))

# ✅ Save `max_length` for inference
joblib.dump(max_length, "max_length.pkl")

# ✅ Convert data to PyTorch tensors
X = torch.tensor(np.array(df["padded"].tolist()), dtype=torch.long)
y = torch.tensor(df["category"].values, dtype=torch.long)

# ✅ Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ PyTorch Dataset & DataLoader
class ExpenseDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = ExpenseDataset(X_train, y_train)
test_dataset = ExpenseDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# ✅ Define Neural Network Model (With More Layers for Better Accuracy)
class ExpenseClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes):
        super(ExpenseClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc1 = nn.Linear(embed_dim, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.embedding(x).mean(dim=1)  # Average word embeddings
        x = self.relu(self.fc1(x))
        return self.fc2(x)

# ✅ Initialize Model with correct vocab size
num_classes = len(label_encoder.classes_)
embed_dim = 64  # Increased embedding dimension for better accuracy

model = ExpenseClassifier(vocab_size=vocab_size, embed_dim=embed_dim, num_classes=num_classes)

# ✅ Define Loss & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ✅ Training Loop (Increase Epochs to 20)
epochs = 20
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader)}")

# ✅ Save Model, Vocab, and Label Encoder
torch.save(model.state_dict(), "expense_classifier_pytorch.pth")
joblib.dump(label_encoder, "label_encoder.pkl")

print("✅ PyTorch model, vocab, and label encoder saved!")
