"""
Page 11: Methodology.
Complete architectural documentation, mathematical formulation, and viva reference.
"""

from __future__ import annotations

import streamlit as st
from components import ce_page_header, ce_source_note

def render(ctx: dict):
    st.markdown(
        ce_page_header(
            "Methodology",
            "Technical documentation: system architecture, mathematical formulations, evaluation protocol, and audit guidelines.",
        ),
        unsafe_allow_html=True,
    )

    st.markdown("""
<div style="background:var(--surface);border:1px solid var(--border);border-radius:var(--radius-lg);padding:16px 20px;margin-bottom:24px;box-shadow:var(--shadow-sm);">
  <b style="font-size:14px;display:block;margin-bottom:8px;">Table of Contents</b>
  <div style="display:grid;grid-template-columns:repeat(3, 1fr);gap:8px;font-size:13px;">
    <a href="#1-problem-statement--framing" target="_self">1. Problem Statement & Framing</a>
    <a href="#2-data-telemetry--selection" target="_self">2. Data Telemetry & Selection</a>
    <a href="#3-chronological-splitting-protocol" target="_self">3. Chronological Splitting Protocol</a>
    <a href="#4-feature-engineering--lag--24h-rule" target="_self">4. Feature Engineering & Lag ≥ 24h</a>
    <a href="#5-forecasting-models" target="_self">5. Forecasting Models</a>
    <a href="#6-anomaly-detection-algorithms" target="_self">6. Anomaly Detection Algorithms</a>
    <a href="#7-synthetic-fault-injection--evaluation" target="_self">7. Synthetic Fault Injection</a>
    <a href="#8-interpretability-shap--tagging" target="_self">8. Interpretability: SHAP & Tagging</a>
    <a href="#9-economic--carbon-impact-model" target="_self">9. Economic & Carbon Impact Model</a>
    <a href="#10-system-limitations" target="_self">10. System Limitations</a>
    <a href="#11-reproducibility--quickstart" target="_self">11. Reproducibility & Quickstart</a>
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown(r"""
### 1. Problem Statement & Framing
University campus energy management faces two major inefficiencies:
- **Operational Wastage:** Systems running when buildings should be quiet (HVAC setbacks missed, lighting left on).
- **Equipment & Meter Faults:** Meter communications dropouts (flatlines), calibration drift, and uncharacteristic surges.

A naive fixed threshold ("alert if load > 300 kWh") cannot work because normal load depends strongly on the hour of the day, day of the week, weather, and campus seasonality.
Therefore, this system frames the problem as **two connected ML tasks**:
1. **Supervised Regression:** Learn conditional normal hourly consumption ($E[y_t | X_t]$) based on calendar, weather, and lagged history.
2. **Unsupervised Anomaly Detection:** Hunt deviations in the residual signal $r_t = y_t - \hat{y}_t$. Evaluation is made objective by injecting synthetic faults with known ground truth.

---

### 2. Data Telemetry & Selection
Data is drawn from the **Building Data Genome Project 2 (BDG2)**, specifically site `Fox`.
Fox was algorithmically selected from 19 candidate sites because it provides 56 usable educational facilities across 5 usage types.
Four buildings were chosen to span wide physical scales and operational profiles:
- **Fox_education_Andre:** College laboratory, $17,044\text{ m}^2$, mean load $\sim 454\text{ kWh/h}$.
- **Fox_education_Elvira:** Student union, $13,827\text{ m}^2$, mean load $\sim 188\text{ kWh/h}$.
- **Fox_education_Claire:** College classroom, $8,145\text{ m}^2$, mean load $\sim 60\text{ kWh/h}$.
- **Fox_education_Kendrick:** Auditorium, $1,053\text{ m}^2$, mean load $\sim 12\text{ kWh/h}$ (baseload dominated).

---

### 3. Chronological Splitting Protocol
To strictly prevent temporal data leakage, all splits use **chronological slicing** rather than random shuffling:
- **Training Fold:** 2016-01-01 00:00 → 2017-04-30 23:00 (16 months, earliest)
- **Validation Fold:** 2017-05-01 00:00 → 2017-08-31 23:00 (4 months, middle)
- **Test Fold:** 2017-09-01 00:00 → 2017-12-31 23:00 (4 months, latest)

All imputers, scalers, and anomaly thresholds are learned exclusively on Training or Validation folds. The Test period is never touched during fitting.

---

### 4. Feature Engineering & Lag ≥ 24h Rule
The single most critical design rule in the pipeline:
> **No feature may use electricity information from less than 24 hours prior to prediction time.**

If an autoregressive feature like $\text{kWh}(t-1)$ were used, an ongoing anomaly at hour $t-1$ would be copied into the expectation for hour $t$. The model would "explain away" the fault, causing the residual to collapse to zero exactly when an alarm is needed. With lags $\ge 24\text{h}$, the expectation describes what the building *should* be doing.

---

### 5. Forecasting Models
Five models were trained and benchmarked:
1. **Seasonal Naive $t-24\text{h}$:** Baseload benchmark copying yesterday's load.
2. **Seasonal Naive $t-168\text{h}$:** Weekly benchmark copying last week's same hour.
3. **Ridge Regression:** L2 regularized linear model with median imputation and standard scaling.
4. **Random Forest:** Bagged trees (`n_estimators=300, min_samples_leaf=3, max_features=0.5`).
5. **LightGBM:** Gradient-boosted trees (`n_estimators=600, lr=0.05, num_leaves=31`).

**Acceptance Standard:** ASHRAE Guideline 14 acceptance threshold for hourly whole-building models requires $\text{CV(RMSE)} \le 30\%$ and $|\text{NMBE}| \le 10\%$. Both Random Forest and LightGBM pass comfortably across all buildings.

---

### 6. Anomaly Detection Algorithms
Four complementary detection engines are deployed:
1. **Residual Robust z-Score:**
   $$z_t = \frac{r_t - \text{median}(r)}{\text{MAD}(r) \times 1.4826}$$
   Using median and Median Absolute Deviation (MAD) provides a $50\%$ breakdown point, preventing outliers from distorting the threshold.
2. **Persistence Guard:**
   Requires $k = 2$ flagged hours out of the last $n = 3$ hours to suppress single-hour sensor noise.
3. **Flatline Guard:**
   Flags hours where rolling standard deviation over 4 hours equals zero for positive values, detecting frozen sensor communications.
4. **Isolation Forest:**
   Unsupervised tree ensemble identifying anomalous points via short path lengths on multivariate context features.
5. **Adaptive Hybrid:**
   Maintains a trailing 30-day (720h) median/MAD window with 24h lag, eliminating concept drift false alarms.

---

### 7. Synthetic Fault Injection & Evaluation
Because real building telemetry lacks reliable human fault labels, ground truth is created by injecting 4 realistic failure modes:
- **Spike:** Multiplicative surges ($2.0\times$ to $3.5\times$, 1–2 hours).
- **After-Hours Load:** Nighttime/weekend excess added ($0.4\times$ to $0.9\times$ median load, 6–10 hours).
- **Flatline:** Frozen sensor readings (8–24 hours).
- **Drift:** Gradual upward ramp ($0.15\times$ to $0.30\times$ median load, 48–72 hours).

---

### 8. Interpretability: SHAP & Tagging
- **TreeExplainer SHAP:** Computes exact Shapley feature contributions summing to the predicted load:
  $$\hat{y} = \phi_0 + \sum_{i=1}^M \phi_i$$
- **Rule-Based Episode Tagging:** Consecutive flagged hours are grouped into episodes and assigned actionable explanations (e.g., "After-hours surge", "Meter freeze", "Slow calibration drift").

---

### 9. Economic & Carbon Impact Model
Energy waste is translated directly into operational cost and greenhouse gas emissions:
$$\text{Avoidable Cost (₹)} = \sum \text{excess\_kWh} \times \text{Tariff (₹8.00/kWh)}$$
$$\text{Avoidable Emissions (kg CO}_2\text{)} = \sum \text{excess\_kWh} \times 0.71\text{ kg/kWh}$$
Only positive excess residuals count as waste; negative deviations (consumption drops) are reported as potential outages rather than cost savings.

---

### 10. System Limitations
- Telemetry originates from US site `Fox`; climate and schedule profiles do not automatically generalize to other geographic campuses.
- Lack of academic holiday metadata creates false alarms during autumn semester start.
- Tariffs and carbon factors are configurable placeholders that must be validated against actual campus utility bills.

---

### 11. Reproducibility & Quickstart
```bash
# Clone and prepare environment
git clone https://github.com/Ashira-senthal/campus-energy.git
cd campus-energy
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Download data and execute pipeline
python src/download_data.py
python run_pipeline.py

# Launch Monitoring Console
streamlit run app/app.py
```
""")

    st.markdown(ce_source_note("README.md + PROJECT_GUIDE.md"), unsafe_allow_html=True)
