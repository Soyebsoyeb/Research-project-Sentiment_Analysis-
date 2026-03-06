import os
import re
import ast
import glob
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from sklearn.manifold import TSNE

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.nn.utils.rnn import pad_sequence
import warnings
warnings.filterwarnings('ignore')

# -------------------- Configuration --------------------
class Config:
    # Data paths - will auto-detect
    data_path = None  # Will be set automatically
    input_dir = '/kaggle/input'
    
    # Vocabulary
    max_vocab_size = 150000          # Increased vocabulary size
    min_freq = 3                      # Lower min frequency to include more words
    max_len = 128                      # Increased sequence length
    
    # Model Architecture - Enhanced for better accuracy
    embed_dim = 400                    # Increased embedding dimension
    hidden_dim = 512                    # Increased hidden dimension
    num_layers = 3                       # More layers
    dropout = 0.3                         # Adjusted dropout
    bidirectional = True               # Use BiLSTM
    
    # Training - Optimized for accuracy
    batch_size = 128                    # Batch size
    learning_rate = 0.001
    epochs = 20                          # More epochs for convergence
    clip_grad = 1.0                        # Gradient clipping
    early_stopping_patience = 5
    scheduler_factor = 0.5
    scheduler_patience = 2
    
    # System
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed = 42
    mixed_precision = torch.cuda.is_available()  # Enable only if CUDA available
    num_workers = 2 if torch.cuda.is_available() else 0
    pin_memory = torch.cuda.is_available()
    sample_size = None  # Use full dataset

config = Config()
print(f"Using device: {config.device}")
print(f"Mixed Precision: {config.mixed_precision}")

# -------------------- Auto-detect dataset --------------------
def find_dataset():
    """Automatically find the Sentiment140 dataset in Kaggle input directory."""
    print("\n" + "="*50)
    print("Searching for Sentiment140 dataset...")
    print("="*50)
    
    # List all files in input directory
    if os.path.exists('/kaggle/input'):
        all_files = glob.glob('/kaggle/input/**/*.csv', recursive=True)
        print(f"Found {len(all_files)} CSV files:")
        for f in all_files:
            print(f"  - {f}")
        
        # Look for sentiment140 files
        sentiment_files = [f for f in all_files if 'sentiment' in f.lower() or 'training' in f.lower()]
        
        if sentiment_files:
            config.data_path = sentiment_files[0]
            print(f"\n Selected dataset: {config.data_path}")
            return True
    
    print("\n Could not find Sentiment140 dataset automatically.")
    return False

if not find_dataset():
    raise FileNotFoundError("Sentiment140 dataset not found. Please ensure it's uploaded to Kaggle.")

# -------------------- Reproducibility --------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(config.seed)

# -------------------- Text Preprocessing --------------------
def clean_token(token):
    """Clean individual token."""
    if not isinstance(token, str):
        return ""
    token = token.lower()
    # Keep alphanumeric and basic punctuation that might carry sentiment
    token = re.sub(r'[^a-z0-9!?.]', '', token)
    return token

def parse_text_list(text_str):
    """Convert string like "['word1', 'word2']" to list of words."""
    if not isinstance(text_str, str):
        return []
    try:
        # Remove any extra quotes if present and evaluate
        return ast.literal_eval(text_str)
    except:
        try:
            # If literal_eval fails, try cleaning the string
            cleaned = text_str.strip('[]').replace("'", "").split(', ')
            return [word.strip("'") for word in cleaned]
        except:
            return []

# -------------------- Load and Prepare Dataset --------------------
print("\n" + "="*50)
print("Loading Sentiment140 dataset (custom format)...")
print("="*50)

# Read CSV with two columns (no header)
df_raw = pd.read_csv(
    config.data_path,
    encoding='latin-1',
    header=None,
    names=['target', 'text'],
    on_bad_lines='skip'
)

print(f"Raw shape: {df_raw.shape}")
print("First 5 rows:")
print(df_raw.head())

# Check if first row is a header (contains "text" in second column)
if df_raw.iloc[0, 1] == 'text':
    print("First row appears to be a header. Removing it.")
    df_raw = df_raw.iloc[1:].reset_index(drop=True)

print(f"\nAfter potential header removal: {df_raw.shape}")

# Convert target to numeric, coercing errors
df_raw['target'] = pd.to_numeric(df_raw['target'], errors='coerce')

# Drop rows where target is NaN
df_raw = df_raw.dropna(subset=['target']).reset_index(drop=True)
print(f"After dropping invalid targets: {df_raw.shape}")

print("\nParsing text lists...")
tqdm.pandas(desc="Parsing")
df_raw['tokens'] = df_raw['text'].progress_apply(parse_text_list)

# Filter out rows with empty token lists
df_raw = df_raw[df_raw['tokens'].apply(len) > 0].reset_index(drop=True)
print(f"After filtering empty token lists: {len(df_raw)} samples")

print("Cleaning tokens...")
df_raw['tokens'] = df_raw['tokens'].progress_apply(
    lambda tokens: [clean_token(t) for t in tokens if clean_token(t)]
)

# Filter again after cleaning
df_raw = df_raw[df_raw['tokens'].apply(len) > 0].reset_index(drop=True)
print(f"After cleaning tokens: {len(df_raw)} samples")

# Show a sample
print("\nSample processed data:")
print(df_raw[['target', 'tokens']].head(3))

# Check target distribution
print("\nTarget value counts (before binarization):")
print(df_raw['target'].value_counts().sort_index())

# Binarize targets - For Sentiment140, 0 = negative, 4 = positive
# Map 4 to 1 (positive) and 0 to 0 (negative)
df_raw['target_binary'] = (df_raw['target'] == 4).astype(int)
print(f"\nBinarizing (0=negative, 4=positive):")
print(df_raw['target_binary'].value_counts())

# Use the binary target for training
df = df_raw.copy()
df['target'] = df['target_binary']

# Show token statistics
token_lengths = [len(tokens) for tokens in df['tokens']]
print(f"\nToken statistics:")
print(f"Min tokens: {min(token_lengths)}")
print(f"Max tokens: {max(token_lengths)}")
print(f"Mean tokens: {np.mean(token_lengths):.2f}")
print(f"Median tokens: {np.median(token_lengths):.2f}")

# -------------------- Build Vocabulary --------------------
print("\n" + "="*50)
print("Building vocabulary...")
print("="*50)

all_tokens = [token for tokens in df['tokens'] for token in tokens]
word_counts = Counter(all_tokens)
print(f"Total tokens: {len(all_tokens):,}")
print(f"Unique tokens: {len(word_counts):,}")

# Keep words with frequency >= min_freq, up to max_vocab_size (leaving room for special tokens)
vocab_words = [word for word, count in word_counts.most_common(config.max_vocab_size - 2) 
               if count >= config.min_freq]

word2idx = {word: idx+2 for idx, word in enumerate(vocab_words)}  # reserve 0 for PAD, 1 for UNK
word2idx['<PAD>'] = 0
word2idx['<UNK>'] = 1
idx2word = {idx: word for word, idx in word2idx.items()}
vocab_size = len(word2idx)

print(f"Vocabulary size: {vocab_size:,}")
if len(all_tokens) > 0:
    coverage = sum(word_counts[word] for word in vocab_words) / len(all_tokens) * 100
    print(f"Coverage: {coverage:.2f}% of all tokens")
print(f"Most common words: {vocab_words[:20]}")

# -------------------- Encode Sequences --------------------
def encode(tokens, word2idx, max_len):
    """Convert tokens to indices, truncate to max_len."""
    ids = [word2idx.get(token, word2idx['<UNK>']) for token in tokens[:max_len]]
    return torch.tensor(ids, dtype=torch.long)

print("\nEncoding sequences...")
encoded = [encode(tokens, word2idx, config.max_len) for tokens in tqdm(df['tokens'])]
targets = torch.tensor(df['target'].values, dtype=torch.float32)

# Pad sequences to max length in batch
print("Padding sequences...")
encoded_padded = pad_sequence(encoded, batch_first=True)
print(f"Encoded shape: {encoded_padded.shape}")

# Create dataset
dataset = TensorDataset(encoded_padded, targets)

# -------------------- Train/Val/Test Split --------------------
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

print(f"\nDataset split:")
print(f"Train samples: {len(train_dataset):,} ({len(train_dataset)/len(dataset)*100:.1f}%)")
print(f"Val samples: {len(val_dataset):,} ({len(val_dataset)/len(dataset)*100:.1f}%)")
print(f"Test samples: {len(test_dataset):,} ({len(test_dataset)/len(dataset)*100:.1f}%)")

# -------------------- DataLoader --------------------
def collate_batch(batch):
    """Batch collation function."""
    texts, labels = zip(*batch)
    texts = torch.stack(texts)
    labels = torch.stack(labels)
    return texts, labels

train_loader = DataLoader(
    train_dataset, 
    batch_size=config.batch_size, 
    shuffle=True,
    collate_fn=collate_batch, 
    num_workers=config.num_workers, 
    pin_memory=config.pin_memory
)

val_loader = DataLoader(
    val_dataset, 
    batch_size=config.batch_size, 
    shuffle=False,
    collate_fn=collate_batch, 
    num_workers=config.num_workers, 
    pin_memory=config.pin_memory
)

test_loader = DataLoader(
    test_dataset, 
    batch_size=config.batch_size, 
    shuffle=False,
    collate_fn=collate_batch, 
    num_workers=config.num_workers, 
    pin_memory=config.pin_memory
)

# -------------------- Improved Model Definition --------------------
class AttentionLayer(nn.Module):
    """Attention mechanism for LSTM outputs."""
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention_weights = nn.Linear(hidden_dim, 1, bias=False)
        
    def forward(self, lstm_outputs):
        # lstm_outputs: (batch, seq_len, hidden_dim)
        attention_scores = self.attention_weights(lstm_outputs).squeeze(-1)
        attention_weights = torch.softmax(attention_scores, dim=1)
        
        # Apply attention weights
        weighted_output = torch.sum(lstm_outputs * attention_weights.unsqueeze(-1), dim=1)
        return weighted_output

class SentimentModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout, bidirectional):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embedding_dropout = nn.Dropout(0.2)
        
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        lstm_output_dim = hidden_dim * (2 if bidirectional else 1)
        
        # Add attention mechanism
        self.attention = AttentionLayer(lstm_output_dim)
        
        # Classifier with multiple layers
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_output_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Embedding layer with dropout
        emb = self.embedding(x)
        emb = self.embedding_dropout(emb)
        
        # LSTM layer
        lstm_out, (hidden, cell) = self.lstm(emb)
        
        # Use attention mechanism
        attended_features = self.attention(lstm_out)
        
        # Classifier
        out = self.classifier(attended_features)
        
        return self.sigmoid(out).squeeze()

# Initialize model
model = SentimentModel(
    vocab_size=vocab_size,
    embed_dim=config.embed_dim,
    hidden_dim=config.hidden_dim,
    num_layers=config.num_layers,
    dropout=config.dropout,
    bidirectional=config.bidirectional
).to(config.device)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nModel parameters:")
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# -------------------- Loss, Optimizer, Scheduler --------------------
criterion = nn.BCELoss()

# Use AdamW with weight decay for better regularization
optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-5)

# OneCycleLR scheduler for better convergence
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=config.learning_rate,
    epochs=config.epochs,
    steps_per_epoch=len(train_loader),
    pct_start=0.3,
    anneal_strategy='cos'
)

# Mixed Precision (if available)
scaler = torch.cuda.amp.GradScaler() if config.mixed_precision else None

# -------------------- Training Functions --------------------
def train_epoch(model, loader, optimizer, criterion, scheduler, device, clip_grad, scaler=None):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc="Training", leave=False)
    for texts, labels in pbar:
        texts, labels = texts.to(device), labels.to(device)
        optimizer.zero_grad()

        if scaler:
            with torch.cuda.amp.autocast():
                outputs = model(texts)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(texts)
            loss = criterion(outputs, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()
        
        scheduler.step()

        # Statistics
        total_loss += loss.item() * texts.size(0)
        preds = (outputs >= 0.5).float()
        correct += (preds == labels).sum().item()
        total += texts.size(0)

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}', 
            'acc': f'{correct/total:.4f}'
        })

    return total_loss / total, correct / total

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for texts, labels in tqdm(loader, desc="Evaluating", leave=False):
            texts, labels = texts.to(device), labels.to(device)
            outputs = model(texts)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item() * texts.size(0)
            probs = outputs.cpu().numpy()
            preds = (probs >= 0.5).astype(int)
            
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary'
    )
    
    return avg_loss, acc, precision, recall, f1, all_preds, all_labels, all_probs

# -------------------- Early Stopping --------------------
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_score):
        if self.best_score is None:
            self.best_score = val_score
        elif val_score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_score
            self.counter = 0
        return self.early_stop

early_stopping = EarlyStopping(patience=config.early_stopping_patience)

# -------------------- Training Loop --------------------
print("\n" + "="*50)
print("Starting Training...")
print("="*50)

history = {
    'train_loss': [], 'train_acc': [],
    'val_loss': [], 'val_acc': [],
    'val_precision': [], 'val_recall': [], 'val_f1': []
}

best_val_acc = 0
best_epoch = 0

for epoch in range(1, config.epochs + 1):
    print(f"\n{'='*40}")
    print(f"Epoch {epoch}/{config.epochs}")
    print(f"{'='*40}")
    
    # Train
    train_loss, train_acc = train_epoch(
        model, train_loader, optimizer, criterion, scheduler,
        config.device, config.clip_grad, scaler
    )
    
    # Validate
    val_loss, val_acc, val_prec, val_rec, val_f1, _, _, _ = evaluate(
        model, val_loader, criterion, config.device
    )
    
    # Save history
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    history['val_precision'].append(val_prec)
    history['val_recall'].append(val_rec)
    history['val_f1'].append(val_f1)
    
    # Print results
    print(f"\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
    print(f"Precision: {val_prec:.4f} | Recall: {val_rec:.4f} | F1: {val_f1:.4f}")
    
    if val_acc > 0.95:
        print(f" Target achieved! Validation accuracy > 95%")
    
    # Save best model based on validation accuracy
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_epoch = epoch
        # Save with safe globals
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
            'val_loss': val_loss,
            'config': config,
            'word2idx': word2idx,
            'idx2word': idx2word
        }, 'best_model.pt')
        print(f" Saved best model (epoch {epoch}) with val_acc: {val_acc:.4f}")
    
    # Early stopping
    if early_stopping(val_loss):
        print(f"\n Early stopping triggered at epoch {epoch}")
        break
    
    if val_acc > 0.96:  # Early stop if we exceed 96%
        print(f"\n Reached high accuracy! Stopping early.")
        break

print(f"\nTraining completed! Best model from epoch {best_epoch} with val_acc: {best_val_acc:.4f}")

# -------------------- Training Curves --------------------
if len(history['train_loss']) > 0:
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Loss curves
    axes[0, 0].plot(history['train_loss'], label='Train Loss', marker='o')
    axes[0, 0].plot(history['val_loss'], label='Val Loss', marker='o')
    if best_epoch > 0:
        axes[0, 0].axvline(x=best_epoch-1, color='r', linestyle='--', label='Best Model')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Accuracy curves
    axes[0, 1].plot(history['train_acc'], label='Train Acc', marker='o')
    axes[0, 1].plot(history['val_acc'], label='Val Acc', marker='o')
    if best_epoch > 0:
        axes[0, 1].axvline(x=best_epoch-1, color='r', linestyle='--', label='Best Model')
    axes[0, 1].axhline(y=0.95, color='g', linestyle='--', label='95% Target')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Training and Validation Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # F1 Score
    axes[1, 0].plot(history['val_f1'], label='Val F1', marker='o', color='purple')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].set_title('Validation F1 Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Precision & Recall
    axes[1, 1].plot(history['val_precision'], label='Precision', marker='o')
    axes[1, 1].plot(history['val_recall'], label='Recall', marker='o')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_title('Precision and Recall')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.show()

# -------------------- Test Evaluation --------------------
print("\n" + "="*50)
print("Evaluating on Test Set")
print("="*50)

# Load best model with safe globals
if os.path.exists('best_model.pt'):
    # Method 1: Add Config to safe globals (recommended)
    from torch.serialization import add_safe_globals
    add_safe_globals([Config])
    checkpoint = torch.load('best_model.pt', map_location=config.device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded best model from epoch {checkpoint['epoch']} with val_acc: {checkpoint['val_acc']:.4f}")

# Evaluate on test set
test_loss, test_acc, test_prec, test_rec, test_f1, test_preds, test_labels, test_probs = evaluate(
    model, test_loader, criterion, config.device
)

print(f"\nTest Results:")
print(f"Loss: {test_loss:.4f}")
print(f"Accuracy: {test_acc:.4f}")
print(f"Precision: {test_prec:.4f}")
print(f"Recall: {test_rec:.4f}")
print(f"F1-Score: {test_f1:.4f}")

if test_acc >= 0.95:
    print("🎉 SUCCESS! Achieved 95%+ test accuracy!")
else:
    print(f"📊 Current test accuracy: {test_acc:.2%}. Need {max(0, 0.95-test_acc):.2%} more to reach 95%.")

# Classification Report
print("\nDetailed Classification Report:")
print(classification_report(test_labels, test_preds, target_names=['Negative', 'Positive']))

# Confusion Matrix
cm = confusion_matrix(test_labels, test_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Negative', 'Positive'], 
            yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title(f'Confusion Matrix (Accuracy: {test_acc:.3f})')
plt.show()

# -------------------- ROC Curve --------------------
from sklearn.metrics import roc_curve, auc, roc_auc_score

fpr, tpr, _ = roc_curve(test_labels, test_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

print(f"ROC-AUC Score: {roc_auc:.4f}")

# -------------------- Sample Predictions --------------------
print("\n" + "="*50)
print("Sample Predictions")
print("="*50)

def predict_sentiment(text, model, word2idx, max_len, device):
    """Predict sentiment for a single text."""
    model.eval()
    
    # Simple tokenization by splitting
    words = text.lower().split()
    cleaned_words = [clean_token(w) for w in words if clean_token(w)]
    
    # Encode
    encoded = encode(cleaned_words, word2idx, max_len)
    encoded = encoded.unsqueeze(0).to(device)  # Add batch dimension
    
    # Predict
    with torch.no_grad():
        output = model(encoded)
        prob = output.item()
        pred = "Positive" if prob >= 0.5 else "Negative"
    
    return pred, prob

# Test with sample texts
sample_texts = [
    "I love this movie! It's absolutely fantastic and wonderful",
    "This is terrible, worst experience ever, I hate it",
    "The product is okay, nothing special but works fine",
    "Absolutely amazing! Best purchase ever made",
    "Waste of money, completely useless, very disappointed",
    "Not bad, could be better but acceptable",
    "This is the best thing I've ever bought! Highly recommended!",
    "I'm so angry right now, terrible customer service",
    "Pretty good actually, surprised me",
    "Disappointing quality, expected much better"
]

print("\nPredictions:")
print("-" * 60)
for text in sample_texts:
    pred, prob = predict_sentiment(text, model, word2idx, config.max_len, config.device)
    confidence = prob if prob >= 0.5 else 1 - prob
    print(f"Text: {text[:50]}...")
    print(f"Sentiment: {pred} (confidence: {confidence:.3f})")
    print("-" * 60)

# -------------------- Error Analysis --------------------
print("\n" + "="*50)
print("Error Analysis")
print("="*50)

# Get indices of misclassified examples
test_indices = test_dataset.indices
misclassified_indices = [i for i, (true, pred) in enumerate(zip(test_labels, test_preds)) if true != pred]

if len(misclassified_indices) > 0:
    print(f"Total misclassified: {len(misclassified_indices)} out of {len(test_labels)} ({len(misclassified_indices)/len(test_labels)*100:.2f}%)")
    
    # Show some misclassified examples
    print("\nSample misclassified examples:")
    print("-" * 60)
    for idx in misclassified_indices[:10]:  # Show first 10
        true_label = "Positive" if test_labels[idx] == 1 else "Negative"
        pred_label = "Positive" if test_preds[idx] == 1 else "Negative"
        prob = test_probs[idx]
        
        # Get the original text (need to reconstruct from tokens)
        data_idx = test_indices[idx]
        tokens = df.iloc[data_idx]['tokens']
        text_preview = ' '.join(tokens[:20]) + ('...' if len(tokens) > 20 else '')
        
        print(f"Text: {text_preview}")
        print(f"True: {true_label} | Pred: {pred_label} | Prob: {prob:.3f}")
        print("-" * 60)

# -------------------- Save Model and Results --------------------
print("\n" + "="*50)
print("Saving Results")
print("="*50)

# Save model in different formats with safe loading in mind
torch.save(model.state_dict(), 'sentiment140_model_weights.pth')

# Save complete model with Config class safely handled
torch.save({
    'model_state_dict': model.state_dict(),
    'word2idx': word2idx,
    'idx2word': idx2word,
    'config_dict': {k: v for k, v in config.__dict__.items() if not k.startswith('__')},
    'test_accuracy': test_acc,
    'test_f1': test_f1,
    'test_auc': roc_auc
}, 'sentiment140_complete_model.pt')

print(" Model saved successfully!")
print("- sentiment140_model_weights.pth (weights only)")
print("- sentiment140_complete_model.pt (complete model with metadata)")

# Save results to CSV
results_df = pd.DataFrame({
    'Metric': ['Test Loss', 'Test Accuracy', 'Test Precision', 'Test Recall', 'Test F1', 'ROC-AUC'],
    'Value': [test_loss, test_acc, test_prec, test_rec, test_f1, roc_auc]
})
results_df.to_csv('test_results.csv', index=False)
print(" Results saved to test_results.csv")

# Save predictions
predictions_df = pd.DataFrame({
    'True_Label': test_labels,
    'Predicted_Label': test_preds,
    'Probability': test_probs
})
predictions_df.to_csv('test_predictions.csv', index=False)
print(" Predictions saved to test_predictions.csv")

print("\n" + "="*50)
print("Training Complete!")
print("="*50)
if test_acc >= 0.95:
    print("CONGRATULATIONS! Achieved 95%+ test accuracy! ")
else:
    print(f"Final test accuracy: {test_acc:.2%}")
