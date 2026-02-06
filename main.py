"""
Integrated Gradients for Essay Analysis
Two-Layer Architecture: Section-Level + Essay-Level Analysis
"""

import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModel, AutoTokenizer
from captum.attr import IntegratedGradients
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

# Base model to use
BASE_MODEL_NAME = "distilbert-base-uncased"

# File paths
ESSAYS_FILE = "data/EssayAnalysisData.xlsx"
OUTPUT_FILE = "data/IG_Results2.xlsx"

# Sections to analyze
SECTIONS = ['Hypothesis', 'Argument', 'Counter_Argument', 'Conclusion']
SECTION_CODES = {'Hypothesis': 'H', 'Argument': 'A', 'Counter_Argument': 'CA', 'Conclusion': 'C'}

# Layer 1: Section-Level Axes (analyzed per section)
SECTION_AXIS_NAMES = [
    "Emotional_Sum",    # -1 to 1
    "Modality",         # -1 to 1
    "Stance",           # -1 to 1
    "Justice"           # -1 to 1
]

# Layer 2: Essay-Level Axes (analyzed on full essay)
ESSAY_AXIS_NAMES = [
    "Concern"           # 0 to 1
]

# ============================================================================
# STEP 1: CREATE MULTI-OUTPUT MODELS
# ============================================================================

class SectionLevelModel(nn.Module):
    """
    Section-Level Model: Analyzes individual sections
    Outputs: 4 scores (Emotional_Sum, Modality, Stance, Justice)
    """
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        hidden_size = base_model.config.hidden_size

        # Regression head: outputs 4 values (one per section axis)
        self.regressor = nn.Linear(hidden_size, len(SECTION_AXIS_NAMES))

    def forward(self, input_ids, attention_mask=None):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        logits = self.regressor(cls_output)

        # Apply tanh to constrain outputs to [-1, 1] range
        scores = torch.tanh(logits)

        return scores


class EssayLevelModel(nn.Module):
    """
    Essay-Level Model: Analyzes entire essay
    Outputs: 1 score (Concern)
    """
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        hidden_size = base_model.config.hidden_size

        # Regression head: outputs 1 value (Concern)
        self.regressor = nn.Linear(hidden_size, len(ESSAY_AXIS_NAMES))

    def forward(self, input_ids, attention_mask=None):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        logits = self.regressor(cls_output)

        # Apply sigmoid to constrain output to [0, 1] range
        scores = torch.sigmoid(logits)

        return scores

# ============================================================================
# STEP 2: LOAD DATA
# ============================================================================

print("Loading essay data...")
essays_df = pd.read_excel(ESSAYS_FILE, sheet_name='ESSAYS')
print(f"   Loaded {len(essays_df)} essays\n")

# Get first essay for demonstration
essay_id = essays_df.iloc[0]['Essay_ID']
student_id = essays_df.iloc[0]['Student_ID']
print(f"Analyzing Essay: {essay_id} (Student: {student_id})\n")

# ============================================================================
# STEP 3: LOAD MODELS
# ============================================================================

print(f"Loading base model: {BASE_MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

# Create two separate models
print("   Creating Section-Level Model...")
base_model_section = AutoModel.from_pretrained(BASE_MODEL_NAME)
section_model = SectionLevelModel(base_model_section)
section_model.eval()

print("   Creating Essay-Level Model...")
base_model_essay = AutoModel.from_pretrained(BASE_MODEL_NAME)
essay_model = EssayLevelModel(base_model_essay)
essay_model.eval()

# Initialize with random weights (simulating trained models)
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
# STEP 4: HELPER FUNCTIONS
# ============================================================================

def predict_section_scores(text, model):
    """Get section-level predictions for 4 axes"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

    with torch.no_grad():
        scores = model(inputs['input_ids'], inputs['attention_mask'])
        scores = scores.squeeze().tolist()

    return {axis: score for axis, score in zip(SECTION_AXIS_NAMES, scores)}


def predict_essay_scores(text, model):
    """Get essay-level predictions for 1 axis"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

    with torch.no_grad():
        scores = model(inputs['input_ids'], inputs['attention_mask'])
        scores = scores.squeeze().tolist()

    # Handle single value
    if not isinstance(scores, list):
        scores = [scores]

    return {axis: score for axis, score in zip(ESSAY_AXIS_NAMES, scores)}


def get_integrated_gradients_for_axis(text, axis_index, model, is_section_level=True):
    """Calculate IG attributions for a specific axis"""

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    # Get input embeddings
    input_embeddings = model.base_model.embeddings.word_embeddings(input_ids)
    baseline_embeddings = torch.zeros_like(input_embeddings)

    # Custom forward function
    def forward_func(embeddings):
        base_outputs = model.base_model(
            inputs_embeds=embeddings,
            attention_mask=attention_mask
        )

        cls_output = base_outputs.last_hidden_state[:, 0, :]
        logits = model.regressor(cls_output)

        # Apply appropriate activation
        if is_section_level:
            scores = torch.tanh(logits)  # [-1, 1]
        else:
            scores = torch.sigmoid(logits)  # [0, 1]

        return scores[:, axis_index]

    # Initialize and calculate IG
    ig = IntegratedGradients(forward_func)
    attributions = ig.attribute(
        inputs=input_embeddings,
        baselines=baseline_embeddings,
        n_steps=50,
        internal_batch_size=1
    )

    # Get per-token scores
    attr_scores = attributions.sum(dim=-1).squeeze(0).detach().numpy()
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    return tokens, attr_scores


def summarize_attributions(tokens, attr_scores, top_k=10):
    """Get top contributing tokens"""
    filtered = [(tok, score) for tok, score in zip(tokens, attr_scores)
                if tok not in ['[CLS]', '[SEP]', '[PAD]', '<s>', '</s>', '<pad>']]

    sorted_attrs = sorted(filtered, key=lambda x: abs(x[1]), reverse=True)
    return sorted_attrs[:top_k]

# ============================================================================
# STEP 5: ANALYZE SECTION-LEVEL (Layer 1)
# ============================================================================

print("=" * 80)
print("LAYER 1: SECTION-LEVEL ANALYSIS")
print("=" * 80)

section_results = []

for section in SECTIONS:
    section_text = essays_df.iloc[0][section]

    if pd.isna(section_text) or section_text.strip() == "":
        print(f"\n⚠  Skipping {section} - No text found")
        continue

    print(f"\n{'=' * 80}")
    print(f"SECTION: {section}")
    print(f"{'=' * 80}")
    print(f"Text preview: {section_text[:150]}...")

    # Get predictions
    print(f"\nSection-level predictions:")
    scores = predict_section_scores(section_text, section_model)

    for axis_name, score in scores.items():
        print(f"   {axis_name:<20}: {score:>6.3f}")

    # Calculate IG for each axis
    print(f"\nCalculating Integrated Gradients...")

    for axis_idx, axis_name in enumerate(SECTION_AXIS_NAMES):
        print(f"\n   Analyzing {axis_name}...")

        tokens, attr_scores = get_integrated_gradients_for_axis(
            section_text, axis_idx, section_model, is_section_level=True
        )

        top_contributors = summarize_attributions(tokens, attr_scores, top_k=10)

        print(f"   Top 5 tokens:")
        print(f"      {'Token':<20} {'Attribution':>12}")
        print(f"      {'-'*20} {'-'*12}")
        for token, score in top_contributors[:5]:
            print(f"      {token:<20} {score:>12.6f}")

        # Store results
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
# STEP 6: ANALYZE ESSAY-LEVEL (Layer 2)
# ============================================================================

print(f"\n{'=' * 80}")
print("LAYER 2: ESSAY-LEVEL ANALYSIS")
print("=" * 80)

# Concatenate all sections
full_essay_text = " ".join([
    essays_df.iloc[0]['Hypothesis'],
    essays_df.iloc[0]['Argument'],
    essays_df.iloc[0]['Counter_Argument'],
    essays_df.iloc[0]['Conclusion']
])

print(f"\nFull essay length: {len(full_essay_text)} characters")
print(f"Text preview: {full_essay_text[:150]}...")

# Get predictions
print(f"\nEssay-level predictions:")
essay_scores = predict_essay_scores(full_essay_text, essay_model)

for axis_name, score in essay_scores.items():
    print(f"   {axis_name:<20}: {score:>6.3f}")

# Calculate IG for essay-level axis
print(f"\nCalculating Integrated Gradients for Concern...")

essay_results = []

for axis_idx, axis_name in enumerate(ESSAY_AXIS_NAMES):
    tokens, attr_scores = get_integrated_gradients_for_axis(
        full_essay_text, axis_idx, essay_model, is_section_level=False
    )

    top_contributors = summarize_attributions(tokens, attr_scores, top_k=15)

    print(f"\n   Top 10 tokens for {axis_name}:")
    print(f"      {'Token':<20} {'Attribution':>12}")
    print(f"      {'-'*20} {'-'*12}")
    for token, score in top_contributors[:10]:
        print(f"      {token:<20} {score:>12.6f}")

    # Store results
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
# STEP 7: EXPORT RESULTS
# ============================================================================

print(f"\n{'=' * 80}")
print("EXPORTING RESULTS")
print(f"{'=' * 80}")

# Combine all results
all_results = section_results + essay_results
results_df = pd.DataFrame(all_results)

# Export to Excel
with pd.ExcelWriter(OUTPUT_FILE, engine='openpyxl') as writer:

    # Sheet 1: Complete Token Attributions
    results_df.to_excel(writer, sheet_name='Token_Attributions', index=False)

    # Sheet 2: Summary Statistics
    summary_df = results_df.groupby(['Layer', 'Section', 'Axis']).agg({
        'Attribution_Score': ['mean', 'std', 'min', 'max'],
        'Predicted_Score': 'first'
    }).round(4)
    summary_df.to_excel(writer, sheet_name='Summary')

    # Sheet 3: Top Contributors
    top_per_layer_section_axis = []

    # Section-level top contributors
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

    # Essay-level top contributors
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

    # Sheet 4: Section-Level Scores (SCORINGS format)
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

    # Sheet 5: Essay-Level Scores
    essay_scoring = []
    for axis in ESSAY_AXIS_NAMES:
        data = results_df[
            (results_df['Layer'] == 'Essay') &
            (results_df['Axis'] == axis)
        ]
        if len(data) > 0:
            essay_scoring.append({
                'Essay_ID': essay_id,
                'Axis': axis,
                'Score': data.iloc[0]['Predicted_Score']
            })

    if essay_scoring:
        essay_scores_df = pd.DataFrame(essay_scoring)
        essay_scores_df.to_excel(writer, sheet_name='Essay_Scores', index=False)

print(f"✓ Results exported to: {OUTPUT_FILE}")
print(f"\n Output Structure:")
print(f"  - Token_Attributions: Complete attribution data (both layers)")
print(f"  - Summary: Statistics per layer/section/axis")
print(f"  - Top_Contributors: Most influential tokens")
print(f"  - Section_Scores: Layer 1 predictions (matches SCORINGS format)")
print(f"  - Essay_Scores: Layer 2 predictions (Concern)")

print(f"\n{'=' * 80}")
print("ANALYSIS COMPLETE")
print(f"{'=' * 80}")
print(f"""
 Summary:
   - Section-Level Analysis: {len(SECTIONS)} sections × {len(SECTION_AXIS_NAMES)} axes = {len(SECTIONS) * len(SECTION_AXIS_NAMES)} IG runs
   - Essay-Level Analysis: {len(ESSAY_AXIS_NAMES)} axis = {len(ESSAY_AXIS_NAMES)} IG run
   - Total IG Calculations: {len(SECTIONS) * len(SECTION_AXIS_NAMES) + len(ESSAY_AXIS_NAMES)}
   
""")