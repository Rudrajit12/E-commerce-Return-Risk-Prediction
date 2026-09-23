# E-Commerce Return Intelligence System

> **End-to-end AI decision system for predicting, explaining, and acting
> on e-commerce return risk.**

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B)
![SHAP](https://img.shields.io/badge/Explainability-SHAP-purple)
![ML](https://img.shields.io/badge/ML-Scikit--learn%20%7C%20XGBoost-orange)

## 1. Project Overview

E-commerce returns create direct operational and financial costs through
reverse logistics, handling, refunds, restocking, and lost margin. This
project builds an end-to-end **Return Intelligence System** that
predicts return risk at order placement and turns that prediction into
an explainable business action.

The system: - predicts return probability immediately after order
placement; - explains predictions using SHAP; - converts risk into a
cost-aware action; - estimates expected intervention value; - generates
human-readable LLM explanations; - exposes the workflow through FastAPI;
and - provides an interactive Streamlit dashboard.

**Business question:** Which newly placed orders have elevated return
risk, why does the model believe that, and is intervention economically
worthwhile under the configured assumptions?

## 2. Architecture

``` text
Order
  ↓
Feature Pipeline
  ↓
ML Model ──────────────┐
  ↓                    │
Return Probability     │
  ↓                    │
Decision Engine        │
  ↓                    │
Business Action        │
  ↓                    │
SHAP Explainability ───┘
  ↓
Structured Decision Context
  ↓
LLM Explanation
  ↓
FastAPI
  ↓
Streamlit Dashboard
```

The LLM is an **explanation layer, not the decision-maker**. The model
determines probability, the decision engine determines action, and SHAP
provides model-attribution evidence.

## 3. Dataset

The project uses a reproducible synthetic e-commerce dataset designed to
resemble an order-level retail prediction problem.

  Attribute                                        Value
  ------------------ -----------------------------------
  Orders                                          40,000
  Customers                                        8,000
  Products                                         2,500
  Period                                   Jan--Dec 2025
  Return rate                                   \~21.46%
  Target                                      `returned`
  Prediction point     Immediately after order placement

The synthetic data contains realistic missingness and behavioral
patterns so the complete pipeline can be demonstrated without exposing
proprietary customer or retailer data.

**Limitation:** observed relationships are demonstration/modeling
behavior, not real-world causal evidence.

## 4. Data Quality and Leakage Prevention

The validation pipeline checks schema, required columns, duplicate order
IDs, missing values, dates, numeric ranges, customer-history
consistency, categorical values, target validity, and post-event
leakage.

Explicitly blocked post-event fields include:

``` text
return_reason
return_requested_date
refund_amount
actual_delivery_days
delivered_date
post_delivery_rating
refund_date
return_date
```

This keeps the prediction point aligned with information actually
available when an order is placed.

## 5. Feature Engineering

The model uses 17 features.

**Numeric:** `product_price`, `product_rating`,
`product_historical_return_rate`, `customer_age`,
`customer_previous_orders`, `customer_previous_returns`,
`customer_return_rate`, `quantity`, `discount_pct`,
`delivery_distance_km`, `expected_delivery_days`, `order_month`,
`order_day_of_week`.

**Categorical:** `category`, `payment_type`, `discount_bucket`,
`price_bucket`.

Categorical variables are one-hot encoded and numerical variables are
standardized inside the model pipeline. Identifiers are retained for
traceability but are not predictive features.

## 6. Temporal Evaluation

Because this is a future-order prediction problem, the final
model-selection workflow uses chronological splits rather than a random
train/test split:

``` text
Train       → Jan–Aug 2025
Validation  → Sep 2025
Test        → Oct–Dec 2025
```

  Split            Rows   Return Rate
  ------------ -------- -------------
  Train          26,703        21.33%
  Validation      3,297        20.50%
  Test           10,000        22.13%

## 7. Model Development and Selection

Two candidates were evaluated:

-   **Logistic Regression** --- transparent, interpretable baseline.
-   **XGBoost** --- nonlinear tree-based benchmark.

The production candidate was not chosen simply because one algorithm was
more complex. Models were evaluated using a validation period and the
same business decision framework.

**Selected production candidate: Logistic Regression** with a frozen
decision threshold of **0.21** return probability.

On the untouched final test period, the selected model achieved
approximately:

  Metric          Logistic Regression
  ------------- ---------------------
  ROC-AUC                       0.672
  PR-AUC                        0.361
  Precision                    32.36%
  Recall                       61.77%
  F1                            0.425
  Log Loss                      0.497
  Brier Score                   0.161

The default 0.50 threshold produced very low recall, demonstrating why
an operational threshold should be aligned with the business objective
rather than assumed to be 0.50.

## 8. Business Decision Engine

The frozen policy is:

``` text
Probability >= 0.21 → HIGH   → INTERVENE
0.10–0.20           → MEDIUM → MONITOR
Probability < 0.10  → LOW    → NO_ACTION
```

Demonstration economics:

``` text
Intervention cost = ₹100/order
Avoided return cost = ₹800/return
Intervention effectiveness = 60%
```

``` text
Expected avoided return cost
    = P(return) × effectiveness × avoided return cost

Net expected value
    = Expected avoided return cost - intervention cost
```

These are scenario assumptions, not measured causal effects or realized
savings.

On the 10,000-order test population, the decision engine produced:

  Action        Orders
  ----------- --------
  INTERVENE      4,224
  MONITOR        4,220
  NO_ACTION      1,556

## 9. SHAP Explainability

SHAP is used for global and individual prediction explanations.

Important model-attribution drivers included customer return rate,
product historical return rate, Fashion category, product rating,
Footwear category, discount level, price bucket, payment type, discount
percentage, and previous customer returns.

Example high-risk order:

``` text
Order ID: O031068
Return probability: 68.80%
Risk level: HIGH
Action: INTERVENE
```

Important positive contributions included customer return rate, product
historical return rate, previous customer returns, Fashion category,
product rating, and delivery distance. Negative contributions included
UPI payment, medium discount, and premium price bucket.

**Caveat:** SHAP explains how features contributed to the model
prediction. It does not prove that changing a feature would causally
change return behavior.

## 10. LLM Explanation Layer

The architecture is:

``` text
ML Prediction
    ↓
SHAP Factors
    ↓
Decision Engine
    ↓
Structured JSON
    ↓
LLM
    ↓
Human-readable explanation
```

The LLM is constrained from changing probability, risk level, action,
threshold, or business assumptions; inventing facts; treating SHAP as
causal evidence; or presenting modeled savings as realized savings.

This separation demonstrates a key AI engineering pattern:
**deterministic decision logic + generative explanation**.

## 11. FastAPI

Endpoints:

``` text
GET  /health
POST /predict
```

Run locally:

``` bash
uvicorn src.api.main:app --reload
```

Interactive API docs:

``` text
http://127.0.0.1:8000/docs
```

The `/predict` workflow validates the request, generates probability,
applies the frozen policy, computes business value, generates SHAP
explanations, and returns a structured response with the LLM
explanation.

## 12. Streamlit Dashboard

The Streamlit dashboard provides four business-facing areas:

### Executive Overview

-   Test population
-   Observed return rate
-   High-risk orders
-   Intervention volume
-   Modeled net value
-   Probability distribution

### Risk Analysis

-   Category risk
-   Payment-type analysis
-   Discount analysis
-   Risk distribution
-   Highest-risk orders

### Order Simulator

Users can enter order attributes and receive return probability, risk
level, recommended action, expected avoided return cost, intervention
cost, and expected net value.

### AI Explanation

Displays a natural-language explanation grounded in model probability,
decision policy, SHAP contributions, and business assumptions.

Run locally:

``` bash
streamlit run app.py
```

## 13. Project Structure

``` text
ecommerce-return-intelligence/
│
├── app.py
├── requirements.txt
├── README.md
│
├── data/
│   ├── raw/
│   └── processed/
│       ├── model_selection/
│       ├── explainability/
│       ├── decision_engine/
│       └── llm_explanations/
│
├── models/
│   └── model_selection/
│
├── notebooks/
│   └── 01_return_prediction_eda.ipynb
│
├── src/
│   ├── api/
│   ├── data/
│   ├── decision/
│   ├── explainability/
│   ├── features/
│   ├── llm/
│   ├── models/
│   ├── model_selection/
│   └── dashboard/
│       └── core.py
│
└── tests/
    ├── test_api.py
    └── test_dashboard.py
```

## 14. Installation

``` bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd ecommerce-return-intelligence
python -m venv .venv
```

Windows:

``` bash
.venv\Scripts\activate
```

macOS/Linux:

``` bash
source .venv/bin/activate
```

Install dependencies:

``` bash
pip install -r requirements.txt
```

## 15. Run the Pipeline

Generate data:

``` bash
python generate_data.py
```

Validate and clean:

``` bash
python src/data/validate.py
python src/data/cleaning.py
```

Build features and splits:

``` bash
python src/features/build_features.py
python src/data/split_data.py
```

Train baseline and benchmark:

``` bash
python src/models/train_baseline.py
python src/models/train_xgboost.py
```

Run model selection:

``` bash
python src/model_selection/train_candidates.py
python src/model_selection/select_model.py
```

Generate explanations and decisions:

``` bash
python src/explainability/shap_explain.py
python src/decision/decision_engine.py
python src/llm/llm_explainer.py --mode mock
```

Run API:

``` bash
uvicorn src.api.main:app --reload
```

Run dashboard:

``` bash
streamlit run app.py
```

Run tests:

``` bash
python tests/test_api.py
python tests/test_dashboard.py
```

## 16. LLM Configuration

For OpenAI mode, configure environment variables without committing
secrets:

``` text
OPENAI_API_KEY=<your-key>
OPENAI_MODEL=<your-model>
LLM_MODE=openai
```

For development without an API key:

``` text
LLM_MODE=mock
```

Never commit API keys, `.env`, or Streamlit secrets.

## 17. Reproducibility

Random seeds are fixed for synthetic data generation, model training,
SHAP sampling, and mock LLM output. The data pipeline, model-selection
pipeline, SHAP layer, decision engine, and dashboard logic have been
executed and checked for deterministic behavior.

## 18. Engineering Decisions

**Why Logistic Regression?** The project compares a transparent baseline
with XGBoost instead of assuming model complexity equals business value.
Validation-based model selection favored the Logistic Regression
candidate under the configured decision economics.

**Why temporal splitting?** The model predicts future orders, so
chronological evaluation better reflects deployment than a random split.

**Why threshold optimization?** A 0.50 cutoff is not inherently optimal
when intervention has a cost and a correctly targeted intervention has
potential value.

**Why SHAP?** Probability alone is hard for business users to act on.
Feature-level attribution makes the prediction more actionable.

**Why an LLM?** The LLM provides a natural-language interface over
deterministic model and business outputs without replacing the decision
logic.

## 19. Limitations

This is a portfolio-grade prototype rather than a production system.

-   Data is synthetic.
-   Intervention effectiveness is assumed.
-   Avoided return cost is assumed.
-   No real intervention experiment has been conducted.
-   Business-value estimates are modeled expectations.
-   No real-time feature store is implemented.
-   API authentication and authorization are not implemented.
-   Monitoring and drift detection are not implemented.
-   Cloud deployment and CI/CD are future extensions.

## 20. Future Improvements

-   Real retailer/order data integration
-   Real-time feature store
-   Model calibration monitoring
-   Data/model drift detection
-   Automated retraining
-   A/B testing of interventions
-   Return-reason prediction
-   Return-cost prediction
-   Customer lifetime-value-aware intervention policy
-   Reverse-logistics optimization
-   API authentication
-   CI/CD
-   Docker and cloud deployment
-   Production observability

## 21. Portfolio Takeaway

This project demonstrates a complete AI engineering workflow rather than
an isolated ML notebook:

``` text
Business Problem
      ↓
Data Generation & Validation
      ↓
Leakage Prevention
      ↓
Feature Engineering
      ↓
Temporal Model Evaluation
      ↓
Model Selection
      ↓
Decision Policy
      ↓
SHAP Explainability
      ↓
LLM Explanation
      ↓
FastAPI
      ↓
Streamlit Dashboard
```

> **Prediction is only one part of an AI system. The useful system
> connects prediction to explanation, decision-making, and an
> operational interface.**

## 22. Author

**Rudrajit**\
AI / Data Analytics · Machine Learning · AI Automation · Decision
Systems

Built as part of an AI Engineer portfolio focused on practical business
problems using machine learning, explainability, automation, and
generative AI.
