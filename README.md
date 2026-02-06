# Integrated Gradients Essay Analysis System

## Overview

This system implements Integrated Gradients (IG) for analyzing essay texts across x psychological and linguistic dimensions. The analysis evaluates student essays on conflict resolution topics, providing interpretable explanations of which textual elements influence scoring across multiple axes.

## Project Structure

```
EssayAnalysis/
│
├── main.py                      # Main analysis script
├── requirements.txt             # Python dependencies
├── README.md                    # This file
└── data/
    ├── EssayAnalysisData.xlsx   # INPUT: Student essays
    └── IG_Results.xlsx          # OUTPUT: Attribution analysis results
```

## System Requirements

- Python 3.8 or higher
- 8GB RAM minimum
- 2GB free disk space (for model files)

## Installation

### Step 1: Clone or Download Project

Github (To do) 

### Step 2: Create Virtual Environment (Recommended)

```bash
python -m venv .venv1
source .venv1/bin/activate  # On Windows: .venv1\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- PyTorch (deep learning framework)
- Transformers (Hugging Face library)
- Captum (interpretability library)
- Pandas, NumPy, OpenPyXL (data processing)

## Input Data Format

### Excel File: `data/EssayAnalysisData.xlsx`

The input file must contain two sheets:

#### Sheet 1: "ESSAYS"

| Column | Type | Description |
|--------|------|-------------|
| Essay_ID | String | Unique identifier (e.g., "E001") |
| Student_ID | String | Student identifier (e.g., "A00000000") |
| Hypothesis | Text | Student's thesis statement |
| Argument | Text | Supporting arguments section |
| Counter_Argument | Text | Counterargument analysis section |
| Conclusion | Text | Concluding remarks section |

## Running the Analysis

### Basic Usage

```bash
python main.py
```

### Expected Output

The script will:
1. Load essays from Excel
2. Analyze the first essay across four sections
3. Predict scores on five axes per section
4. Calculate Integrated Gradients attributions
5. Export results to `data/IG_Results.xlsx`

### Console Output

```
 Loading essay data...
 Loaded 3 essays

Analyzing Essay: E001 (Student: A00000000)

Loading base model: distilbert-base-uncased
   ✓ Model loaded successfully
   ✓ Configured for 5 axes: Valence_Tone, Stance, Moral_Load, Intensity_Urgency, Certainty_Hedging

================================================================================
STARTING ANALYSIS
================================================================================

SECTION: Hypothesis
Getting model predictions for all 5 axes...
   Valence_Tone        : 0.234
   Stance              : 0.567
   ...

Calculating Integrated Gradients for each axis...
   Analyzing Valence_Tone...
   Top 10 tokens for Valence_Tone:
      Token                 Attribution
      -------------------- ------------
      diplomatic               0.045321
      peace                    0.038912
      ...
```

## Output Structure

### Excel File: `data/IG_Results.xlsx`

The output contains four sheets:

#### Sheet 1: Token_Attributions

Complete token-by-token attribution data.

**Columns:**
- `Essay_ID`: Essay identifier
- `Section`: Essay section (Hypothesis, Argument, Counter_Argument, Conclusion)
- `Axis`: Analysis dimension (Valence_Tone, Stance, Moral_Load, Intensity_Urgency, Certainty_Hedging)
- `Predicted_Score`: Model's prediction for this axis (-1 to +1)
- `Token_Position`: Position of token in sequence (0, 1, 2, ...)
- `Token`: The actual token/word
- `Attribution_Score`: How much this token contributed to the prediction

**Use case:** Deep analysis of which specific words influenced each score.

#### Sheet 2: Summary

Statistical summary per section and axis.

**Columns:**
- `Section`: Essay section
- `Axis`: Analysis dimension
- `Attribution_Score (mean)`: Average attribution across all tokens
- `Attribution_Score (std)`: Standard deviation of attributions
- `Attribution_Score (min)`: Most negative attribution value
- `Attribution_Score (max)`: Most positive attribution value
- `Predicted_Score`: The final predicted score

**Use case:** Quick statistical overview of attribution distributions.

#### Sheet 3: Top_Contributors

Top 10 most influential tokens per section-axis combination.

**Columns:**
- `Section`: Essay section
- `Axis`: Analysis dimension
- `Token`: The token text
- `Attribution_Score`: Magnitude of contribution
- `Predicted_Score`: Final score for reference

**Use case:** Identifying key words that drove predictions. Most useful for interpretation and reporting.

#### Sheet 4: Predicted_Scores (EXTRA FOR PERSONAL FORMATTING)

Clean summary matching the SCORINGS template format.

**Columns:**
- `Essay_ID`: Essay identifier
- `Section`: Essay section code (H, A, CA, C)
- `Valence_Tone`: Score for emotional tone dimension
- `Stance`: Score for position-taking dimension
- `Moral_Load`: Score for ethical/moral language dimension
- `Intensity_Urgency`: Score for rhetorical force dimension
- `Certainty_Hedging`: Score for epistemic certainty dimension

**Use case:** Clean format for reporting final scores to stakeholders.

## Analysis Dimensions

### Axes of Analysis

1. **Valence_Tone** (-1 to +1)
   - Measures emotional polarity of language
   - -1: Highly negative, critical, pessimistic
   - 0: Neutral, factual
   - +1: Highly positive, optimistic

2. **Stance** (-1 to +1)
   - Indicates position relative to hypothesis
   - -1: Against (contra)
   - 0: Neutral/ambivalent
   - +1: In favor (pro)

3. **Moral_Load** (-1 to +1)
   - Measures ethical/moral judgment language
   - -1: Strong moral condemnation
   - 0: Descriptive, non-judgmental
   - +1: Strong moral approval

4. **Intensity_Urgency** (-1 to +1)
   - Reflects rhetorical force and urgency
   - -1: Calm, moderate, diplomatic
   - 0: Neutral intensity
   - +1: Vehement, urgent, emphatic

5. **Certainty_Hedging** (-1 to +1)
   - Indicates epistemic confidence
   - -1: Speculative, hedging language
   - 0: Conditional statements
   - +1: Categorical, certain statements

## Interpreting Results

### Attribution Scores

**Positive Attribution (+):** Token pushed prediction toward higher values (more positive)

**Negative Attribution (-):** Token pushed prediction toward lower values (more negative)

**Magnitude:** Larger absolute values indicate stronger influence

### Example Interpretation

If `Stance = 0.78` for Hypothesis section:

```
Top tokens:
- "intervention" (+0.25)  ← Strongly pro-intervention
- "diplomatic" (+0.15)    ← Moderately pro-intervention
- "necessary" (+0.13)     ← Moderately pro-intervention
```

**Interpretation:** The model assigned a high Stance score because it identified strongly supportive language like "intervention," "diplomatic," and "necessary."

### Common Token Types

**Special Tokens:**
- `[CLS]`: Sentence-level representation token (BERT architecture)
- `[SEP]`: Separator token
- `[PAD]`: Padding token

**Subword Tokens:**
- `multi` + `##lateral` = "multilateral"
- `veri` + `##fia` + `##ble` = "verifiable"
- The `##` prefix indicates continuation of previous token

**Low Attribution Tokens:**
- Many tokens (stopwords, articles) contribute minimally
- Attribution scores < 0.01 are common and expected
- Only a few keywords typically drive predictions

## Technical Details

### Model Architecture

```
Input Text
    ↓
Tokenization (BERT WordPiece)
    ↓
Embeddings Layer (768-dimensional)
    ↓
DistilBERT Transformer (6 layers)
    ↓
[CLS] Token Representation
    ↓
Linear Regression Head (x outputs)
    ↓
Tanh Activation (constrain to [-1, 1])
    ↓
Axis Scores
```

### Integrated Gradients Process

For each axis (5 total per section):

1. **Baseline Creation:** Zero embeddings (represents "blank" input)
2. **Path Interpolation:** Create 50 intermediate points from baseline to input
3. **Gradient Calculation:** Compute gradients at each interpolation step
4. **Integration:** Sum gradients along the path
5. **Attribution Extraction:** Aggregate to token-level scores

**Computational Cost:** 50 forward passes per axis = 250 forward passes per section

### Configuration Parameters

Located in `main.py` lines 20-30:

```python
BASE_MODEL_NAME = "distilbert-base-uncased"  # Base transformer model
ESSAYS_FILE = "data/EssayAnalysisData.xlsx"  # Input file path
OUTPUT_FILE = "data/IG_Results.xlsx"         # Output file path
SECTIONS = ['Hypothesis', 'Argument', 'Counter_Argument', 'Conclusion']
AXIS_NAMES = ["Valence_Tone", "Stance", "Moral_Load", "Intensity_Urgency", "Certainty_Hedging"]
```

**Integrated Gradients Parameters** (line 162):

```python
n_steps=50              # Number of interpolation steps (20-100 typical)
internal_batch_size=1   # Batch size for gradient calculation
```

## Citation and References

Original Paper of the Integrated Gradients methodology:

Sundararajan, M., Taly, A., & Yan, Q. (2017). Axiomatic attribution for deep networks. *Proceedings of the 34th International Conference on Machine Learning*, 70, 3319-3328. https://arxiv.org/abs/1703.01365

For technical issues or questions about the implementation, consult:

- Captum Documentation: https://captum.ai/
- Hugging Face Transformers: https://huggingface.co/docs/transformers/
- PyTorch Documentation: https://pytorch.org/docs/
