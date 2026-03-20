"""
train_essay_model.py
Trains essay-level model (ConcernSelf, ConcernOthers, Modality)
Uses full essay text with Longformer
"""

import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import LongformerModel, LongformerPreTrainedModel, AutoTokenizer, AdamW
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

# ============================================================================
# CONFIGURATION
# ============================================================================

SEED = 42
MODEL_NAME = "allenai/longformer-base-4096"
EXCEL_PATH = "data/training/graded_essays.xlsx"
SAVE_DIR = "models"

MAX_LENGTH = 2048
BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 4
NUM_EPOCHS = 15
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)

# ============================================================================
# DATA LOADING
# ============================================================================

def load_essay_data(excel_path):
    """
    Load essay-level training data

    Expected Excel structure (Sheet 2 - Essay Level):
    ID | Hypothesis | Arguments | CounterArguments | Conclusion |
    ConcernSelf | ConcernOthers | Modality

    Creates full essay text by concatenating all 4 sections
    Returns DataFrame with one row per essay
    """
    df = pd.read_excel(excel_path, sheet_name=1)

    essays_data = []

    for idx, row in df.iterrows():
        essay_id = row['ID']

        full_text = f"[H] {row['Hypothesis']}\n\n[A] {row['Arguments']}\n\n[CA] {row['CounterArguments']}\n\n[C] {row['Conclusion']}"

        essays_data.append({
            'essay_id': essay_id,
            'text': full_text,
            'concern_self': row['ConcernSelf'] if pd.notna(row.get('ConcernSelf')) else 0.5,
            'concern_others': row['ConcernOthers'] if pd.notna(row.get('ConcernOthers')) else 0.5,
            'modality': row['Modality'] if pd.notna(row.get('Modality')) else 0.0
        })

    return pd.DataFrame(essays_data)

# ============================================================================
# DATASET
# ============================================================================

class EssayDataset(Dataset):
    """
    PyTorch Dataset for essay-level training

    Each item contains:
    - Tokenized full essay (input_ids, attention_mask)
    - Global attention mask
    - Labels (3 scores: concern_self, concern_others, modality)
    """
    def __init__(self, dataframe, tokenizer, max_length=2048):
        self.data = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        encoding = self.tokenizer(
            row['text'],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        global_attention_mask = torch.zeros_like(encoding['input_ids'])
        global_attention_mask[0, 0] = 1

        labels = torch.tensor([
            row['concern_self'],
            row['concern_others'],
            row['modality']
        ], dtype=torch.float32)

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'global_attention_mask': global_attention_mask.squeeze(0),
            'labels': labels
        }

# ============================================================================
# MODEL
# ============================================================================

class LongformerEssayRegressor(LongformerPreTrainedModel):
    """
    Longformer model with regression head for essay-level scoring

    Architecture:
    1. Longformer base (pre-trained, mostly frozen)
    2. Dropout layer
    3. Linear layer: 768 dims -> 3 outputs
    4. Mixed activation:
       - concern_self, concern_others: sigmoid [0, 1]
       - modality: tanh [-1, 1]

    Outputs 3 scores: concern_self, concern_others, modality
    """
    def __init__(self, config):
        super().__init__(config)
        self.longformer = LongformerModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.regressor = nn.Linear(config.hidden_size, 3)
        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, global_attention_mask=None, labels=None):
        outputs = self.longformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask
        )

        pooled = outputs.last_hidden_state[:, 0, :]
        pooled = self.dropout(pooled)
        raw_logits = self.regressor(pooled)

        concern_self = torch.sigmoid(raw_logits[:, 0:1])
        concern_others = torch.sigmoid(raw_logits[:, 1:2])
        modality = torch.tanh(raw_logits[:, 2:3])

        logits = torch.cat([concern_self, concern_others, modality], dim=1)

        loss = None
        if labels is not None:
            loss = nn.functional.mse_loss(logits, labels)

        return {'loss': loss, 'logits': logits}

# ============================================================================
# TRAINING
# ============================================================================

def train_model(model, train_loader, val_loader, optimizer, scaler, epochs, device):
    """
    Training loop with validation
    Same process as section model but for essay-level data
    """
    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        optimizer.zero_grad()

        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for i, batch in enumerate(progress_bar):
            batch = {k: v.to(device) for k, v in batch.items()}

            with autocast():
                outputs = model(**batch)
                loss = outputs['loss'] / GRAD_ACCUM_STEPS

            scaler.scale(loss).backward()

            if (i + 1) % GRAD_ACCUM_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            train_loss += outputs['loss'].item()
            progress_bar.set_postfix({'loss': f"{outputs['loss'].item():.4f}"})

        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validation'):
                batch = {k: v.to(device) for k, v in batch.items()}

                with autocast():
                    outputs = model(**batch)
                    val_loss += outputs['loss'].item()

        avg_val_loss = val_loss / len(val_loader)

        print(f'\nEpoch {epoch+1}: Train={avg_train_loss:.4f}, Val={avg_val_loss:.4f}')

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f'{SAVE_DIR}/essay_model.pt')
            print('  ✓ Model saved!')

    return model

def main():
    """
    Main training pipeline using LOEO cross-validation
    """
    os.makedirs(SAVE_DIR, exist_ok=True)

    print(f"Using device: {device}")
    print("Loading data...")

    df_essays = load_essay_data(EXCEL_PATH)
    print(f"Loaded {len(df_essays)} essays")

    essay_ids = sorted(df_essays['essay_id'].unique())
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("\nStarting LOEO Cross-Validation...")

    for held_out_id in essay_ids:
        print(f"\nFOLD: Held-out Essay = {held_out_id}")

        train_df = df_essays[df_essays['essay_id'] != held_out_id].reset_index(drop=True)
        val_df = df_essays[df_essays['essay_id'] == held_out_id].reset_index(drop=True)

        train_dataset = EssayDataset(train_df, tokenizer, MAX_LENGTH)
        val_dataset = EssayDataset(val_df, tokenizer, MAX_LENGTH)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

        model = LongformerEssayRegressor.from_pretrained(MODEL_NAME)
        model.to(device)

        for param in model.longformer.parameters():
            param.requires_grad = False
        for param in model.longformer.encoder.layer[-1].parameters():
            param.requires_grad = True

        optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scaler = GradScaler()

        train_model(model, train_loader, val_loader, optimizer, scaler, NUM_EPOCHS, device)

    print("\n✓ Training complete!")

if __name__ == '__main__':
    main()