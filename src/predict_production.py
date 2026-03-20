"""
predict_production.py
Production inference using trained Longformer models
Reads new essays, generates predictions, exports results
"""

import pandas as pd
import torch
from transformers import AutoTokenizer
from pathlib import Path
import sys

sys.path.append('src')
from train_section_model import LongformerSectionRegressor
from train_essay_model import LongformerEssayRegressor

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_NAME = "allenai/longformer-base-4096"
SECTION_MODEL_PATH = "models/section_model.pt"
ESSAY_MODEL_PATH = "models/essay_model.pt"

INPUT_FILE = "data/production/new_essays.xlsx"
OUTPUT_FILE = "data/results/predictions.xlsx"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================================
# LOAD MODELS
# ============================================================================

print(f"Using device: {device}")
print("Loading models...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

section_model = LongformerSectionRegressor.from_pretrained(MODEL_NAME)
section_model.load_state_dict(torch.load(SECTION_MODEL_PATH, map_location=device))
section_model.to(device)
section_model.eval()

essay_model = LongformerEssayRegressor.from_pretrained(MODEL_NAME)
essay_model.load_state_dict(torch.load(ESSAY_MODEL_PATH, map_location=device))
essay_model.to(device)
essay_model.eval()

print("✓ Models loaded\n")


# ============================================================================
# PREDICTION FUNCTIONS
# ============================================================================

@torch.no_grad()
def predict_section(text):
    encoding = tokenizer(text, truncation=True, padding='max_length', max_length=1024, return_tensors='pt')

    batch = {k: v.to(device) for k, v in encoding.items()}
    global_attention_mask = torch.zeros_like(batch['input_ids'])
    global_attention_mask[0, 0] = 1
    batch['global_attention_mask'] = global_attention_mask

    outputs = section_model(**batch)
    scores = outputs['logits'].cpu().numpy().squeeze()

    return {
        'emotional_sum': float(scores[0]),
        'modality': float(scores[1]),
        'stance': float(scores[2]),
        'justice': float(scores[3])
    }


@torch.no_grad()
def predict_essay(text):
    encoding = tokenizer(text, truncation=True, padding='max_length', max_length=2048, return_tensors='pt')

    batch = {k: v.to(device) for k, v in encoding.items()}
    global_attention_mask = torch.zeros_like(batch['input_ids'])
    global_attention_mask[0, 0] = 1
    batch['global_attention_mask'] = global_attention_mask

    outputs = essay_model(**batch)
    scores = outputs['logits'].cpu().numpy().squeeze()

    return {
        'concern_self': float(scores[0]),
        'concern_others': float(scores[1])
    }


# ============================================================================
# MAIN INFERENCE
# ============================================================================

print("Loading essays...")
df = pd.read_excel(INPUT_FILE, sheet_name='ESSAYS')
print(f"Loaded {len(df)} essays\n")

results = []

for idx, row in df.iterrows():
    essay_id = row['Essay_ID']
    print(f"Processing {essay_id}...")

    sections = {
        'H': row['Hypothesis'],
        'A': row['Argument'],
        'CA': row['Counter_Argument'],
        'C': row['Conclusion']
    }

    section_scores = {}
    for sec_code, text in sections.items():
        if pd.notna(text) and str(text).strip():
            section_scores[sec_code] = predict_section(str(text))

    full_text = f"[H] {sections['H']}\n\n[A] {sections['A']}\n\n[CA] {sections['CA']}\n\n[C] {sections['C']}"
    essay_scores = predict_essay(full_text)

    result_row = {'Essay_ID': essay_id}

    for sec in ['H', 'A', 'CA', 'C']:
        if sec in section_scores:
            result_row[f'{sec}_EmotionalSum'] = section_scores[sec]['emotional_sum']
            result_row[f'{sec}_Stance'] = section_scores[sec]['stance']
            result_row[f'{sec}_Justice'] = section_scores[sec]['justice']

    result_row['Modality'] = section_scores['H']['modality'] if 'H' in section_scores else 0.0
    result_row['ConcernSelf'] = essay_scores['concern_self']
    result_row['ConcernOthers'] = essay_scores['concern_others']

    results.append(result_row)

results_df = pd.DataFrame(results)
results_df.to_excel(OUTPUT_FILE, index=False)

print(f"\n✓ Predictions saved to {OUTPUT_FILE}")