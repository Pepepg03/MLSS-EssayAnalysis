"""
Integrated Gradients for Essay Analysis
Two-Layer Architecture: Section-Level + Essay-Level Analysis
Uses Longformer for long document processing
"""

import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from transformers import LongformerModel, LongformerTokenizer
from captum.attr import IntegratedGradients
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_MODEL_NAME = "allenai/longformer-base-4096"

ESSAYS_FILE = "data/production/essays_to_score.xlsx"
OUTPUT_FILE = "data/results/IG_Results.xlsx"

SECTIONS = ['hypothesis', 'arguments', 'counterarguments', 'conclusion']
SECTION_CODES = {'hypothesis': 'H', 'arguments': 'A', 'counterarguments': 'CA', 'conclusion': 'C'}

SECTION_AXIS_NAMES = [
    "emotional_sum",
    "modality",
    "stance",
    "justice"
]

ESSAY_AXIS_NAMES = [
    "concern_own",
    "concern_others"
]

MAX_LENGTH = 2048


# ============================================================================
# MODEL ARCHITECTURES
# ============================================================================

class SectionLevelModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        hidden_size = base_model.config.hidden_size
        self.regressor = nn.Linear(hidden_size, len(SECTION_AXIS_NAMES))

    def forward(self, input_ids, attention_mask=None):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        logits = self.regressor(cls_output)
        scores = torch.tanh(logits)
        return scores


class EssayLevelModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        hidden_size = base_model.config.hidden_size
        self.regressor = nn.Linear(hidden_size, len(ESSAY_AXIS_NAMES))

    def forward(self, input_ids, attention_mask=None):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        logits = self.regressor(cls_output)
        scores = torch.sigmoid(logits)
        return scores


# ============================================================================
# LOAD DATA
# ============================================================================

print("📂 Loading essay data...")

df = pd.read_excel(ESSAYS_FILE, sheet_name='Hoja de Evaluación', header=1)

df_clean = df.rename(columns={
    'id': 'essay_id',
    'hypothesis': 'hypothesis',
    'arguments': 'arguments',
    'counterarguments': 'counterarguments',
    'conclusion': 'conclusion'
})

print(f"   Loaded {len(df_clean)} essays\n")

essay_id = df_clean.iloc[0]['essay_id']
print(f"🎯 Analyzing Essay: {essay_id}\n")

# ============================================================================
# LOAD MODELS
# ============================================================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}\n")

print(f"Loading base model: {BASE_MODEL_NAME}")
tokenizer = LongformerTokenizer.from_pretrained(BASE_MODEL_NAME)

print("   Creating Section-Level Model...")
base_model_section = LongformerModel.from_pretrained(BASE_MODEL_NAME)
section_model = SectionLevelModel(base_model_section)
section_model.eval()
section_model.to(device)

print("   Creating Essay-Level Model...")
base_model_essay = LongformerModel.from_pretrained(BASE_MODEL_NAME)
essay_model = EssayLevelModel(base_model_essay)
essay_model.eval()
essay_model.to(device)

torch.manual_seed(42)
with torch.no_grad():
    nn.init.xavier_uniform_(section_model.regressor.weight)
    nn.init.zeros_(section_model.regressor.bias)
    nn.init.xavier_uniform_(essay_model.regressor.weight)
    nn.init.zeros_(essay_model.regressor.bias)

print("   ✓ Models loaded successfully")
print(f"   ✓ Section-Level: {len(SECTION_AXIS_NAMES)} axes ({', '.join(SECTION_AXIS_NAMES)})")
print(f"   ✓ Essay-Level: {len(ESSAY_AXIS_NAMES)} axes ({', '.join(ESSAY_AXIS_NAMES)})\n")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def predict_section_scores(text, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        scores = model(inputs['input_ids'], inputs['attention_mask'])
        scores = scores.squeeze().cpu().tolist()

    return {axis: score for axis, score in zip(SECTION_AXIS_NAMES, scores)}


def predict_essay_scores(text, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        scores = model(inputs['input_ids'], inputs['attention_mask'])
        scores = scores.squeeze().cpu().tolist()

    if not isinstance(scores, list):
        scores = [scores]

    return {axis: score for axis, score in zip(ESSAY_AXIS_NAMES, scores)}


def get_integrated_gradients_for_axis(text, axis_index, model, is_section_level=True):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    input_embeddings = model.base_model.embeddings.word_embeddings(input_ids)
    baseline_embeddings = torch.zeros_like(input_embeddings)

    def forward_func(embeddings):
        base_outputs = model.base_model(
            inputs_embeds=embeddings,
            attention_mask=attention_mask
        )

        cls_output = base_outputs.last_hidden_state[:, 0, :]
        logits = model.regressor(cls_output)

        if is_section_level:
            scores = torch.tanh(logits)
        else:
            scores = torch.sigmoid(logits)

        return scores[:, axis_index]

    ig = IntegratedGradients(forward_func)
    attributions = ig.attribute(
        inputs=input_embeddings,
        baselines=baseline_embeddings,
        n_steps=50,
        internal_batch_size=1
    )

    attr_scores = attributions.sum(dim=-1).squeeze(0).detach().cpu().numpy()
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].cpu())

    return tokens, attr_scores


def summarize_attributions(tokens, attr_scores, top_k=10):
    filtered = [(tok, score) for tok, score in zip(tokens, attr_scores)
                if tok not in ['<s>', '</s>', '<pad>', '<mask>', 'Ġ']]

    sorted_attrs = sorted(filtered, key=lambda x: abs(x[1]), reverse=True)
    return sorted_attrs[:top_k]


# ============================================================================
# ANALYZE SECTION-LEVEL
# ============================================================================

print("=" * 80)
print("LAYER 1: SECTION-LEVEL ANALYSIS")
print("=" * 80)

section_results = []

for section in SECTIONS:
    section_text = df_clean.iloc[0][section]

    if pd.isna(section_text) or str(section_text).strip() == "":
        print(f"\n⚠️  Skipping {section} - No text found")
        continue

    section_text = str(section_text)

    print(f"\n{'=' * 80}")
    print(f"SECTION: {section}")
    print(f"{'=' * 80}")
    print(f"Text preview: {section_text[:150]}...")

    print(f"\n📊 Section-level predictions:")
    scores = predict_section_scores(section_text, section_model)

    for axis_name, score in scores.items():
        print(f"   {axis_name:<20}: {score:>6.3f}")

    print(f"\n🔍 Calculating Integrated Gradients...")

    for axis_idx, axis_name in enumerate(SECTION_AXIS_NAMES):
        print(f"\n   Analyzing {axis_name}...")

        tokens, attr_scores = get_integrated_gradients_for_axis(
            section_text, axis_idx, section_model, is_section_level=True
        )

        top_contributors = summarize_attributions(tokens, attr_scores, top_k=10)

        print(f"   Top 5 tokens:")
        print(f"      {'Token':<20} {'Attribution':>12}")
        print(f"      {'-' * 20} {'-' * 12}")
        for token, score in top_contributors[:5]:
            print(f"      {token:<20} {score:>12.6f}")

        for i, (token, score) in enumerate(zip(tokens, attr_scores)):
            section_results.append({
                'Essay_ID': essay_id,
                'Layer': 'Section',
                'Section': section,
                'Axis': axis_name,
                'Predicted_Score': scores[axis_name],
                'Token_Position': i,
                'Token': token,
                'Attribution_Score': float(score)
            })

# ============================================================================
# ANALYZE ESSAY-LEVEL
# ============================================================================

print(f"\n{'=' * 80}")
print("LAYER 2: ESSAY-LEVEL ANALYSIS")
print("=" * 80)

full_essay_text = " ".join([
    str(df_clean.iloc[0]['hypothesis']),
    str(df_clean.iloc[0]['arguments']),
    str(df_clean.iloc[0]['counterarguments']),
    str(df_clean.iloc[0]['conclusion'])
])

print(f"\nFull essay length: {len(full_essay_text)} characters")
print(f"Text preview: {full_essay_text[:150]}...")

print(f"\nEssay-level predictions:")
essay_scores = predict_essay_scores(full_essay_text, essay_model)

for axis_name, score in essay_scores.items():
    print(f"   {axis_name:<20}: {score:>6.3f}")

print(f"\nCalculating Integrated Gradients for Concern...")

essay_results = []

for axis_idx, axis_name in enumerate(ESSAY_AXIS_NAMES):
    tokens, attr_scores = get_integrated_gradients_for_axis(
        full_essay_text, axis_idx, essay_model, is_section_level=False
    )

    top_contributors = summarize_attributions(tokens, attr_scores, top_k=15)

    print(f"\n   Top 10 tokens for {axis_name}:")
    print(f"      {'Token':<20} {'Attribution':>12}")
    print(f"      {'-' * 20} {'-' * 12}")
    for token, score in top_contributors[:10]:
        print(f"      {token:<20} {score:>12.6f}")

    for i, (token, score) in enumerate(zip(tokens, attr_scores)):
        essay_results.append({
            'Essay_ID': essay_id,
            'Layer': 'Essay',
            'Section': 'Full_Essay',
            'Axis': axis_name,
            'Predicted_Score': essay_scores[axis_name],
            'Token_Position': i,
            'Token': token,
            'Attribution_Score': float(score)
        })

# ============================================================================
# EXPORT RESULTS
# ============================================================================

print(f"\n{'=' * 80}")
print("EXPORTING RESULTS")
print(f"{'=' * 80}")

all_results = section_results + essay_results
results_df = pd.DataFrame(all_results)

with pd.ExcelWriter(OUTPUT_FILE, engine='openpyxl') as writer:
    results_df.to_excel(writer, sheet_name='Token_Attributions', index=False)

    summary_df = results_df.groupby(['Layer', 'Section', 'Axis']).agg({
        'Attribution_Score': ['mean', 'std', 'min', 'max'],
        'Predicted_Score': 'first'
    }).round(4)
    summary_df.to_excel(writer, sheet_name='Summary')

    top_per_layer_section_axis = []

    for section in SECTIONS:
        for axis in SECTION_AXIS_NAMES:
            data = results_df[
                (results_df['Layer'] == 'Section') &
                (results_df['Section'] == section) &
                (results_df['Axis'] == axis)
                ]
            if len(data) > 0:
                top = data.nlargest(10, 'Attribution_Score')[
                    ['Layer', 'Section', 'Axis', 'Token', 'Attribution_Score', 'Predicted_Score']
                ]
                top_per_layer_section_axis.append(top)

    for axis in ESSAY_AXIS_NAMES:
        data = results_df[
            (results_df['Layer'] == 'Essay') &
            (results_df['Axis'] == axis)
            ]
        if len(data) > 0:
            top = data.nlargest(15, 'Attribution_Score')[
                ['Layer', 'Section', 'Axis', 'Token', 'Attribution_Score', 'Predicted_Score']
            ]
            top_per_layer_section_axis.append(top)

    if top_per_layer_section_axis:
        top_df = pd.concat(top_per_layer_section_axis, ignore_index=True)
        top_df.to_excel(writer, sheet_name='Top_Contributors', index=False)

    section_scoring = []
    for section in SECTIONS:
        data = results_df[
            (results_df['Layer'] == 'Section') &
            (results_df['Section'] == section)
            ]
        if len(data) > 0:
            row = {
                'Essay_ID': essay_id,
                'Section': SECTION_CODES[section]
            }
            for axis in SECTION_AXIS_NAMES:
                axis_data = data[data['Axis'] == axis]
                if len(axis_data) > 0:
                    row[axis] = axis_data.iloc[0]['Predicted_Score']
            section_scoring.append(row)

    if section_scoring:
        section_scores_df = pd.DataFrame(section_scoring)
        section_scores_df.to_excel(writer, sheet_name='Section_Scores', index=False)

    essay_scoring = {
        'Essay_ID': essay_id,
        'concern_own': essay_scores['concern_own'],
        'concern_others': essay_scores['concern_others']
    }

    essay_scores_df = pd.DataFrame([essay_scoring])
    essay_scores_df.to_excel(writer, sheet_name='Essay_Scores', index=False)

print(f"✓ Results exported to: {OUTPUT_FILE}")
print(f"\n Output Structure:")
print(f"  - Token_Attributions: Complete attribution data (both layers)")
print(f"  - Summary: Statistics per layer/section/axis")
print(f"  - Top_Contributors: Most influential tokens")
print(f"  - Section_Scores: Layer 1 predictions (emotional_sum, modality, stance, justice)")
print(f"  - Essay_Scores: Layer 2 predictions (concern_own, concern_others)")

print(f"\n{'=' * 80}")
print("ANALYSIS COMPLETE")
print(f"{'=' * 80}")
print(f"""
📝 Summary:
   - Section-Level Analysis: {len(SECTIONS)} sections × {len(SECTION_AXIS_NAMES)} axes = {len(SECTIONS) * len(SECTION_AXIS_NAMES)} IG runs
   - Essay-Level Analysis: {len(ESSAY_AXIS_NAMES)} axes = {len(ESSAY_AXIS_NAMES)} IG runs
   - Total IG Calculations: {len(SECTIONS) * len(SECTION_AXIS_NAMES) + len(ESSAY_AXIS_NAMES)}

⚠️  Note: Models using random weights for demonstration.
   When trained models are ready, load with:
   section_model.load_state_dict(torch.load('models/section_model.pth'))
   essay_model.load_state_dict(torch.load('models/essay_model.pth'))
""")