# Campus Energy Consumption Intelligence & Anomaly Detection

A reproducible machine-learning system that learns the **normal** hourly electricity
behaviour of four university buildings, forecasts what they *should* be using,
flags hours that deviate, explains each flag in plain English, and prices the
excess energy in ₹ and kg CO₂.

Data: [Building Data Genome Project 2](https://github.com/buds-lab/building-data-genome-project-2)
(BDG2), site `Fox`, four education buildings chosen for coverage and usage
diversity:

| Building | Type | Floor area | Mean load |
|---|---|---|---|
| Fox_education_Andre | College laboratory | 17,044 m² | 454 kWh/h |
| Fox_education_Elvira | Student union | 13,827 m² | 188 kWh/h |
| Fox_education_Claire | College classroom | 8,145 m² | 60 kWh/h |
| Fox_education_Kendrick | Auditorium | 1,053 m² | 12 kWh/h |

## Headline results (real, from `reports/`)

Forecast, test fold (Sep–Dec 2017), mean over buildings:

| Model | MAE (kWh) | RMSE (kWh) | CV(RMSE) % | NMBE % |
|---|---|---|---|---|
| Random forest | 9.60 | 15.54 | **19.32** | 4.37 |
| Ridge regression | 10.53 | 16.55 | 18.97 | 2.20 |
| LightGBM | 10.37 | 16.26 | 20.77 | 6.47 |
| Seasonal naive (t−168 h) | 11.83 | 20.89 | 24.49 | −0.68 |
| Seasonal naive (t−24 h) | 13.97 | 23.47 | 25.99 | −0.53 |

Every learned model beats both seasonal-naive baselines, which is the minimum
bar for the features to be worth anything.

Anomaly detection on injected synthetic faults (spike, after-hours load,
flatline, drift), z = 3.0:

| Detector | Precision | Recall | F1 | False alarms |
|---|---|---|---|---|
| Hybrid (persistence + flatline) | 0.454 | 0.350 | **0.338** | 20.3% |
| Residual (robust z) | 0.346 | 0.375 | 0.315 | 25.3% |
| Residual + persistence | 0.364 | 0.312 | 0.291 | 20.2% |
| Isolation Forest | 0.236 | 0.343 | 0.257 | 35.4% |
| Adaptive hybrid (30-day) | 0.732 | 0.166 | 0.262 | **2.8%** |

Read this as a trade-off, not a league table: the hybrid detects the most faults,
the adaptive variant is the only one with an acceptable alarm rate for daily
operation.

## Quickstart

```bash
python3.11 -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt

python src/download_data.py            # ~186 MB of BDG2 data into data/raw/
python run_pipeline.py                 # ~1 minute: features, models, detectors, figures
python -m streamlit run app/app.py     # dashboard at http://localhost:8501
```

`run_pipeline.py` writes `data/processed/*.parquet` (dashboard inputs),
`models/*.joblib` (fitted models), and `reports/*.csv|json|png` (every table and
figure quoted in `PROJECT_GUIDE.md`).

## Repository layout

```text
data/raw/            BDG2 CSVs (git-ignored, fetched by src/download_data.py)
data/processed/      parquet files read by the dashboard
models/              one .joblib per building (5 models) + Isolation Forest
notebooks/           01 data & EDA, 02 model & detector comparison
src/
  config.py          buildings, splits, thresholds, tariff, CO2 factor
  data_loader.py     load BDG2, filter site/buildings, tidy hourly panel
  features.py        cleaning, calendar, weather, lag>=24h features
  models.py          seasonal naive, Ridge, Random Forest, LightGBM
  anomaly.py         robust z, persistence, flatline guard, Isolation Forest,
                     synthetic anomaly injection
  evaluate.py        MAE/RMSE/CV(RMSE)/NMBE/sMAPE, precision/recall/F1
  explain.py         SHAP, anomaly type tagging, kWh -> ₹ -> CO2
  eda.py             static report figures
  download_data.py   reproducible BDG2 download with size checks
run_pipeline.py      end-to-end pipeline
app/app.py           Streamlit dashboard (6 tabs)
reports/             generated tables and figures
PROJECT_GUIDE.md     full viva preparation document
```

## Assumptions you must verify

- **Tariff ₹8.00/kWh** and **CO₂ factor 0.71 kg/kWh** are placeholders. Check your
  campus bill and the current CEA grid factor; both are editable in the dashboard
  sidebar and in `src/config.py`.
- Holidays follow the **data site's country (US)**, because the BDG2 buildings are
  in the United States. Change `HOLIDAY_COUNTRY` in `src/config.py` to `IN` if you
  point the pipeline at your own campus.

See `PROJECT_GUIDE.md` for the full methodology, results, limitations and a
30-question viva bank.
