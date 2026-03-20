"""
train_section_model.py
Trains section-level model (Emotional_Sum, Stance, Justice)
Uses Longformer-base with LOEO cross-validation
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

MAX_LENGTH = 1024
BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 2
NUM_EPOCHS = 15
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01

SECTION_AXES = ["emotional_sum", "stance", "justice"]
SECTION_MAP = {"h": "h_text", "a": "a_text", "ca": "ca_text", "c": "c_text"}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================================
# REPRODUCIBILITY
# ============================================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)

# ============================================================================
# DATA LOADING
# ============================================================================

def load_section_data(excel_path):
    """
    Load section-level training data

    Expected Excel structure (Sheet 1 - Section Level):
    ID | Hypothesis | Arguments | CounterArguments | Conclusion |
    H_EmotionalSum | A_EmotionalSum | CA_EmotionalSum | C_EmotionalSum |
    H_Stance | A_Stance | CA_Stance | C_Stance |
    H_Justice | A_Justice | CA_Justice | C_Justice

    Returns DataFrame with one row per section (4 rows per essay)
    """
    df = pd.read_excel(excel_path, sheet_name=0)

    sections_data = []

    for idx, row in df.iterrows():
        essay_id = row['ID']

        for section_code, section_name in SECTION_MAP.items():
            text = row[section_name]

            if pd.notna(text) and str(text).strip():
                sections_data.append({
                    'essay_id': essay_id,
                    'section': section_code,
                    'text': str(text).strip(),
                    'emotional_sum': row[f'{section_code}_emotional_sum'] if pd.notna(row.get(f'{section_code}_emotional_sum')) else 0.0,
                    'stance': row[f'{section_code}_stance'] if pd.notna(row.get(f'{section_code}_stance')) else 0.0,
                    'justice': row[f'{section_code}_justice'] if pd.notna(row.get(f'{section_code}_justice')) else 0.0
                })

    return pd.DataFrame(sections_data)

# ============================================================================
# DATASET
# ============================================================================

class SectionDataset(Dataset):
    """
    PyTorch Dataset for section-level training

    Each item contains:
    - Tokenized text (input_ids, attention_mask)
    - Global attention mask (marks first token for Longformer)
    - Labels (3 scores: emotional_sum, stance, justice)
    """
    def __init__(self, dataframe, tokenizer, max_length=1024):
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
            row['emotional_sum'],
            row['stance'],
            row['justice']
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

class LongformerSectionRegressor(LongformerPreTrainedModel):
    """
    Longformer model with regression head for section scoring

    Architecture:
    1. Longformer base (pre-trained, mostly frozen)
    2. Dropout layer
    3. Linear layer: 768 dims -> 3 outputs
    4. Tanh activation (constrains to [-1, 1])

    Outputs 3 scores: emotional_sum, stance, justice
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
        logits = torch.tanh(self.regressor(pooled))

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

    Process per epoch:
    1. Training phase: forward pass, backward pass, weight updates
    2. Validation phase: evaluate on held-out data
    3. Save model if validation loss improves
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

        print(f'\nEpoch {epoch+1}:')
        print(f'  Train Loss: {avg_train_loss:.4f}')
        print(f'  Val Loss: {avg_val_loss:.4f}')

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f'{SAVE_DIR}/section_model.pt')
            print('  ✓ Model saved!')

    return model

# ============================================================================
# MAIN
# ============================================================================

def main():
    """
    Main training pipeline using LOEO (Leave-One-Essay-Out) cross-validation

    LOEO Process:
    1. For each essay in dataset:
       - Use that essay as validation
       - Use all other essays for training
    2. This ensures model never sees validation data during training
    3. Prevents overfitting and gives realistic performance estimate
    """
    os.makedirs(SAVE_DIR, exist_ok=True)

    print(f"Using device: {device}")
    print("Loading data...")

    df_sections = load_section_data(EXCEL_PATH)
    print(f"Loaded {len(df_sections)} section samples from {df_sections['essay_id'].nunique()} essays")

    essay_ids = sorted(df_sections['essay_id'].unique())
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("\nStarting LOEO Cross-Validation...")

    for held_out_id in essay_ids:
        print(f"\n{'='*80}")
        print(f"FOLD: Held-out Essay = {held_out_id}")
        print(f"{'='*80}")

        train_df = df_sections[df_sections['essay_id'] != held_out_id].reset_index(drop=True)
        val_df = df_sections[df_sections['essay_id'] == held_out_id].reset_index(drop=True)

        train_dataset = SectionDataset(train_df, tokenizer, MAX_LENGTH)
        val_dataset = SectionDataset(val_df, tokenizer, MAX_LENGTH)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

        model = LongformerSectionRegressor.from_pretrained(MODEL_NAME)
        model.to(device)

        for param in model.longformer.parameters():
            param.requires_grad = False
        for param in model.longformer.encoder.layer[-1].parameters():
            param.requires_grad = True

        optimizer = AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY
        )

        scaler = GradScaler()

        train_model(model, train_loader, val_loader, optimizer, scaler, NUM_EPOCHS, device)

    print("\n✓ Training complete!")

if __name__ == '__main__':
    main()