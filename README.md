<div align="center">

# ⚡ Agentic AI-Based Dynamic Tariff Optimization
### for EV Charging Networks

<img src="https://img.shields.io/badge/IIT%20Roorkee-Open%20Project%202026-1AB67A?style=for-the-badge&logo=graduation-cap&logoColor=white"/>
<img src="https://img.shields.io/badge/SocBiz%20Club-IITR-2B2B2B?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/XGBoost-Demand%20Prediction-FF6600?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Status-Submitted-1AB67A?style=for-the-badge"/>

<br/>

**Sameer Modi** | Roll No: 23410030 | Int. M.Tech 4th Year | Geological Technology

</div>

---

## 🔍 The Problem

> Static ₹15/kWh EV charging tariffs ignore demand completely.
> Peak hours = congested chargers. Off-peak hours = idle chargers.
> Everyone loses — drivers wait, operators waste capacity, grid strains.

```
Peak hours (15-17 PM at Caltech):     2,572 sessions/hour  →  Chargers overloaded
Off-peak hours (9 AM at Caltech):         67 sessions/hour  →  Chargers idle
```

**The fix:** A 3-agent AI system that predicts demand and adjusts prices dynamically — every hour, every district, automatically.

---

## 🤖 Three-Agent Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   UrbanEV Dataset  ──►  AGENT 1           pred_util            │
│   (2.1M rows)           Demand       ──────────────►           │
│   247 districts         Prediction                             │
│   5-min intervals       (XGBoost)                              │
│                                                                 │
│                                        ┌───────────────────┐   │
│                                        │    AGENT 2        │   │
│                            pred_util ─►│  Tariff Pricing   │   │
│                                        │  ₹10.5/₹15/₹21   │   │
│                                        └────────┬──────────┘   │
│                                                 │ dynamic ₹    │
│                                                 ▼              │
│   ACN Dataset      ──►  AGENT 3  ◄─────── simulation          │
│   (14,884 sessions)     Monitoring                             │
│   Real kWh data         & Learning  ──►  7 KPIs               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📊 Datasets Used

| Dataset | Region | Period | Size | Role |
|:---|:---|:---|:---|:---|
| **ACN-Data** (Caltech) | California, USA 🇺🇸 | Apr–Dec 2018 | 14,884 sessions | Revenue KPIs, Pricing Efficiency |
| **UrbanEV / ST-EVCDP** (Shenzhen) | China 🇨🇳 | Jun–Jul 2022 | 2.1M observations | Demand Prediction, Utilization KPIs |

> ⚠️ **Datasets not included** due to size constraints.
> Download from:
> - ACN-Data → [ev.caltech.edu/dataset](https://ev.caltech.edu/dataset)
> - UrbanEV → ST-EVCDP public repository

### 🔑 Key Data Discoveries

| Dataset | Expected Peak | Actual Peak | Why |
|:---|:---|:---|:---|
| ACN (Caltech) | 8–9 AM (office hours) | **3–5 PM** | Research campus — staff arrive mid-afternoon |
| UrbanEV (Shenzhen) | 5–7 PM (commute) | **Midnight–6 AM** | Electrified taxi fleet recharges at night |

---

## 🧠 Agent 1 — Demand Prediction

**Model:** XGBoost | **Target:** Charger utilization 1 hour ahead

### ⚠️ The Data Leakage Problem We Fixed

| Bug | Root Cause | Fake Result | Fix Applied |
|:---|:---|:---|:---|
| `lag_1` feature | Corr 0.997 with target | R² = 0.99 (meaningless) | Removed entirely |
| 5-min ahead target | Autocorrelation trivial | R² inflated | Changed to 1-hour ahead |
| No baseline | Nothing to compare against | Looks impressive | Computed before any ML |

### 📈 Honest Model Results

| Model | RMSE | MAE | R² (Overall) | R² (Transitions) |
|:---|:---:|:---:|:---:|:---:|
| Persistence Baseline | 0.0915 | 0.0516 | 0.7292 | 0.2860 |
| Random Forest | 0.0627 | 0.0373 | 0.8729 | — |
| **XGBoost** ✅ | **0.0578** | **0.0353** | **0.8917** | **0.7579** |

> 💡 **The honest number:** Overall R² = 0.892 is partially inflated by easy stable periods.
> The real skill is on **transition periods** — where demand is actively changing and pricing decisions matter most.
> XGBoost improves R² from **0.286 → 0.758** on transitions, a genuine **+0.47 gain**.

### 🔧 Features Used (14 total, no lag shorter than 1 hour)

```python
FEATS = [
    # Time
    'hour_of_day', 'slot_in_day',   # slot_in_day: 0-287 (5-min resolution)
    'day_of_week', 'is_weekend',

    # District
    'district_id', 'total_chargers', 'is_cbd',

    # History (minimum lag = 1 hour — no leakage)
    'util_lag_12',   # 1 hour ago
    'util_lag_288',  # 24 hours ago
    'util_lag_576',  # 48 hours ago
    'vol_lag_12', 'vol_lag_288',

    # Trend
    'roll_12',   # 60-min rolling avg
    'roll_36',   # 3-hr rolling avg
]
```

---

## 💰 Agent 2 — Tariff Pricing

**Two implementations:** Rule-based (discrete zones) + Sigmoid (smooth curve)

```
Utilization  0.0 ──────── 0.15 ──────────────── 0.38 ──────── 1.0
                  DISCOUNT      NEUTRAL               SURGE
                  ₹10.5/kWh     ₹15.0/kWh            ₹21.0/kWh
                  (−30%)        (baseline)             (+40%)
                     │                                    │
                     └── P25 of training data             └── P75 of training data
```

> **Why P75/P25 (not hardcoded 0.80/0.30)?**
> Hardcoded thresholds on Shenzhen data → 1% surge slots, 61% discount slots → agent barely ever surges. Broken.
> Data-driven P75/P25 → balanced 25% / 50% / 25% split by construction.

### Sigmoid Variant
Smooth continuous pricing on the same ₹10.5–₹21 range.
No abrupt ₹4.50 price jump at zone boundaries.
Centred at data median (0.25) — not 0.5 which biased 93% of outputs toward discount.

---

## 📉 Agent 3 — Monitoring & Learning

### All 7 Evaluation KPIs

| Metric | Value | Source | How Computed |
|:---|:---:|:---|:---|
| **RMSE** | 0.0578 | UrbanEV | ↓36.8% vs persistence baseline |
| **MAE** | 0.0353 | UrbanEV | Avg error ~3.5% utilization |
| **R² Score** | 0.8917 overall / **+0.47** transitions | UrbanEV | Honest split: stable vs transition |
| **Revenue Gain %** | **+14.19%** (sim) / +5.21% (ACN) | UrbanEV + ACN | ((New−Old)/Old)×100 vs ₹15 baseline |
| **Charger Utilization** | 28.8% → **52.0%→45.6%** at peak | UrbanEV | Before/after surge pricing |
| **Off-Peak Uplift** | **+5.32%** | UrbanEV | Extra sessions in discount-zone slots |
| **Avg Wait Reduction** | **7.56%** (proxy) | UrbanEV | % demand displaced from peak slots |
| **Customer Response Rate** | **7.49%** | ACN + elasticity | Avg demand shift/session (ε = −0.30) |
| **Pricing Efficiency** | **₹15.78/kWh** | ACN real kWh | Was ₹15.00/kWh (+₹0.78) |

### Revenue Comparison

```
Baseline (₹15 flat)   ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓          ₹16,657 L
Rule-based Dynamic    ████████████████████████        ₹19,021 L  (+14.19%)
Sigmoid Dynamic       ███████████████████████▉        ₹19,009 L  (+14.12%)
```

---

## ✅ Honest Limitations

We report limitations explicitly — this is what separates honest data science from misleading reports.

| Limitation | Detail |
|:---|:---|
| **Counterfactual simulation** | All demand shift numbers use elasticity = −0.30 from literature, not measured from real driver behaviour |
| **Charger Utilization AFTER** | Estimated via `adjusted_volume / total_chargers` — not from real sensor data post-deployment |
| **Waiting time is a proxy** | No actual queue data in either dataset — 7.56% is demand displaced, not real queue minutes |
| **Customer Response Rate** | Derived from elasticity formula, not directly observed driver decisions |
| **Region specificity** | Model trained on Shenzhen taxi-pattern. Retraining needed for campus or residential contexts |

---

## 🚀 Future Scope

- **Live deployment** — Replace CSV inputs with real-time OCPP sensor feeds
- **Reinforcement learning** — Frame as MDP with Revenue Gain as reward signal
- **Carbon-aware pricing** — Add grid carbon intensity to tariff function (charge more when grid is dirty)
- **Driver personalisation** — Fleet vs passenger car different elasticity profiles

---

## 📁 Repository Structure

```
📦 EV-Tariff-Optimization-OP2026
 ┣ 📓 Final_Submission.ipynb        ← Complete pipeline (Phase 1–5), all cells executed
 ┣ 📊 EV_Tariff_Optimization.pdf   ← Presentation deck (9 slides)
 ┣ 📄 README.md                     ← This file
```

---

## 🔧 How to Run

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/EV-Tariff-Optimization-OP2026

# 2. Open in Google Colab
# Upload Final_Submission.ipynb to Colab

# 3. Upload your dataset zip and update BASE_PATH in Setup cell
BASE_PATH = 'Datasets OP_26 Analytics'

# 4. Run all cells top to bottom (Runtime → Run all)
```

**Requirements**
```
pandas numpy matplotlib seaborn xgboost scikit-learn
```

---

## 📌 Key Design Decisions at a Glance

```
❌ lag_1 feature removed        → was data leakage (corr 0.997 with target)
✅ 1-hour ahead target          → realistic deployment horizon
✅ P75/P25 data-driven          → hardcoded 0.80/0.30 broken for Shenzhen data
✅ Sigmoid centred at 0.25      → 0.5 centre biased 93% outputs to discount
✅ ACN for revenue KPIs         → real kWhDelivered, not estimated energy
✅ Caltech peak = 15-17 PM      → verified from data, not assumed
✅ Shenzhen peak = 00-06 AM     → taxi fleet recharge pattern, verified from data
✅ Elasticity = −0.30           → EV academic literature, stated as assumption
```

---

<div align="center">

**Open Project 2026 | FinTech & SocBiz Club | IIT Roorkee**

*Dynamic tariff optimization reduces peak congestion, increases off-peak utilization,*
*and improves revenue — without requiring hardware changes to existing infrastructure.*

</div>
