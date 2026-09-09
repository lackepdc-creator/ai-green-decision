# AI Assistant Advice Types and Green Decision-Making

> A four-group randomized controlled experiment examining how different explanation levels of AI assistant advice influence college students' pro-environmental decisions in campus scenarios.

🐍 Python 3.10+ &nbsp;·&nbsp; 📊 statsmodels &nbsp;·&nbsp; 📄 MIT License &nbsp;·&nbsp; ✅ Reproducible

---

## 🌟 Project Highlights

- **RCT Design** — Four-group between-subjects experiment (N=202) with batch randomization across 3 campus scenarios (cafeteria, recycling, study area)
- **Rigorous Statistics** — HC3 robust standard errors, Bootstrap mediation (5000 reps), Bayesian factors, TOST equivalence tests, BH-FDR + Holm multiple-testing correction, HTMT discriminant validity
- **Fully Reproducible** — Fixed random seed (20260902), single-command pipeline, bundled synthetic demo data so anyone can run it end-to-end
- **Transparent Null Results** — Manipulation check failed → no significant main effects → reported honestly with Bayesian evidence for H0

---

## 📋 Project Overview

This project investigates **whether AI assistants can genuinely promote green decision-making** in campus settings. We designed a single-factor four-level between-subjects experiment to compare three types of AI advice against a no-advice control group:

| Group | Condition | Description |
|-------|-----------|-------------|
| **A** | Control | No AI advice |
| **B** | Result-only | AI gives conclusion only |
| **C** | Process-based | AI gives conclusion + reasoning |
| **D** | Example-based | AI gives conclusion + reasoning + real-world case |

### Research Questions

- **H1**: Do AI assistants with explanatory advice (B/C/D) increase green choices vs. no advice (A)?
- **H2**: Are process-based and example-based advice more effective than result-only advice?
- **H3**: Do AI trust, cognitive load, and feasibility perception mediate the advice→decision pathway?

---

## 🔬 Key Findings

> ⚠️ **Demo data shown below.** Real data yields similar patterns (null main effects, weak scene-specific trends).

1. **Manipulation check failed** — participants did not reliably distinguish the three advice types (F = 0.23–1.00, p > .39)
2. **No significant main effects** after covariate adjustment (all BH-corrected p ≥ .87)
3. **No significant mediation** — all Bootstrap 95% CIs included zero
4. **Bayesian evidence** — Example group BF₀₁ ≈ 8.4 (moderate support for H₀)
5. **Exploratory** — Ordinal Logit shows positive shifts in the cafeteria scenario (B: OR ≈ 6.11, C: OR ≈ 12.80), suggesting potential scene-dependent effects worth follow-up

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run with demo data (no real data needed!)

```bash
python analysis/top_journal_analysis.py --demo
```

### 3. Check results

Output goes to `results/`:

```
results/
├── analysis_dataset.csv        # Cleaned analysis dataset
├── tables_full.xlsx            # All results tables (20+ sheets)
├── results_summary.txt         # Text summary of key findings
├── figure1_main_effects.png    # Main effects forest plot + means
├── figure2_mediation.png       # Mediation pathways
├── figure3_heterogeneity.png   # Heterogeneity interactions
├── figure4_corr_heatmap.png    # Correlation matrix
├── figure5_consort.png         # CONSORT flow diagram
└── figure6_manip_check.png     # Manipulation check
```

### 4. Use your own data

```bash
python analysis/top_journal_analysis.py --input your_data.xlsx
```

---

## 📁 Repository Structure

```
ai-green-decision/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Git ignore rules
│
├── analysis/
│   ├── top_journal_analysis.py        # Main analysis pipeline (10 steps)
│   ├── survey_detailed_analysis.py    # Survey-specific detailed analysis
│   ├── randomized_experiment_analysis.py  # Core RCT analysis (compact)
│   └── data_cleaning.py               # Data cleaning utilities
│
├── data/
│   ├── generate_demo_data.py          # Synthetic data generator
│   └── demo_dataset.xlsx              # Bundled demo data (202 rows, 33 cols)
│
├── materials/
│   ├── advice_texts.md                # AI advice texts (4 types × 3 scenarios)
│   └── experimental_scenarios.md      # Three campus scenario descriptions
│
└── results/
    ├── results_summary.txt            # Key results (regenerated on run)
    ├── tables_full.xlsx               # All output tables
    ├── analysis_dataset.csv           # Cleaned data CSV
    └── figure*.png                    # Output figures (6 figures)
```

---

## 🧪 Analysis Pipeline (10 Steps)

The main script (`analysis/top_journal_analysis.py`) runs a full top-tier-journal-level analysis:

| Step | Analysis | Method |
|------|----------|--------|
| 1 | Data cleaning & sample flow | CONSORT 2010 flow diagram |
| 2 | Variable construction | Scale composites, z-score index |
| 3 | Manipulation check | One-way ANOVA + Cohen's d + BH-FDR |
| 4 | Reliability & validity | Cronbach's α (Bootstrap CI), HTMT, AVE/CR |
| 5 | Randomization balance | ANOVA + chi-square + effect sizes |
| 6 | Descriptives & correlations | Pearson r with BH-corrected p-values |
| 7 | Main effects | HC3 OLS, binary Logit (OR), multinomial Logit |
| 8 | Mediation | Hayes PROCESS-style Bootstrap (5000 reps) + parallel mediation |
| 9 | Heterogeneity | Interaction models (exploratory, clearly labeled) |
| 10 | Robustness + Bayes | HC3 vs cluster SE, TOST, Bayes factors, post-hoc power |

---

## 🛠 Tech Stack

- **Python 3.10+**
- **Statistics** — pandas, numpy, scipy, statsmodels, patsy
- **Visualization** — matplotlib, seaborn
- **Reporting** — LaTeX (ctexart) for manuscripts, Excel for result tables
- **Reproducibility** — Fixed random seed 20260902

---

## ⚠️ Limitations

- **Manipulation failure** — participants could not distinguish advice types (p > .39)
- **Small sample** — N=202, control group n=31, observed power = 0.06–0.41
- **Batch randomization** — not individual-level; baseline imbalance detected
- **No pre-registration** — study was not pre-registered on OSF

---

## 🔮 Next Steps

- Redesign manipulation materials with larger text differences and visual reinforcement
- Conduct pilot study (N≈45) to verify manipulation effectiveness
- Pre-register on OSF before formal data collection
- Expand sample to N≈500 (125 per group) for 0.80 power
- Switch to individual-level randomization

---

## 📄 License

This project is licensed under the MIT License — see the code for details.
Experimental materials (`materials/`) are released for academic reuse with attribution.

---

<p align="center">
  <i>Built with reproducible research principles.</i>
</p>
