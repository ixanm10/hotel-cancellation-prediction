# 🏨 Hotel Booking Cancellation Prediction

**CDS 4003 – Data Science for Business Application | Group 3**

A machine learning model that predicts at booking time whether a hotel reservation will be cancelled, so staff can act before the room is lost. It's built on 119,390 bookings from a City Hotel (Lisbon) and a Resort Hotel (Algarve), 2015–2017.

**🚀 Live app:** https://hotel-cancellation-group3.streamlit.app

![ML Pipeline](https://github.com/ixanm10/hotel-cancellation-prediction/actions/workflows/pipeline.yml/badge.svg)

## Team

| Name | Role |
|------|------|
| Amna Nasser Alshamsi | |
| Amna Ali Alhosani | |
| Fatema Ahmed Alyileili | |

## Results (hold-out test set)

| Model | F1 | Recall | ROC-AUC |
|-------|----|--------|---------|
| **Random Forest (tuned)** | **0.707** | **0.786** | **0.896** |

Main drivers of cancellation: lead time, deposit type, country, market segment and special requests.

## Repository structure

```
├── .github/workflows/pipeline.yml   # GitHub Actions: runs the pipeline on every push
├── app/app.py                       # Streamlit app (risk score for a new booking)
├── data/hotel_bookings.csv          # raw dataset
├── docs/data_dictionary.md          # description of every column
├── notebooks/Hotel_Cancellation_Colab.ipynb   # full CRISP-DM analysis (Google Colab)
├── reports/CDS4003_Group3_Project_Report.docx
├── src/pipeline.py                  # clean -> features -> train -> evaluate -> save model
├── tests/test_pipeline.py           # data quality checks
└── requirements.txt
```

## The pipeline

```
data/hotel_bookings.csv
   │  1. clean (missing values, duplicates, impossible values, 'Undefined')
   │  2. build features (total_nights, prev_cancel_rate, country_grp, ...)
   │  3. split 80/20 stratified
   │  4. train 6 models (baseline, LR, NB, DT, RF, GBT) + tuning
   │  5. evaluate on test set (F1, recall, AUC, cost of errors)
   ▼  6. save best model
models/model.joblib  ──►  app/app.py (Streamlit)
```

Every push to `main` makes **GitHub Actions** run the tests, run the pipeline, check that F1 ≥ 0.68 and AUC ≥ 0.85, and upload the trained model and figures as a downloadable artifact (Actions tab → latest run → *Artifacts*).

## Run it yourself

```bash
pip install -r requirements.txt
python src/pipeline.py --quick     # fast (~20 s), no tuning
python src/pipeline.py             # full run with hyperparameter tuning
streamlit run app/app.py           # open the app
```

The notebook can also be opened in Colab directly from GitHub: upload `data/hotel_bookings.csv` to the Colab Files panel, then run all cells.

## Data source

Antonio, N., de Almeida, A., & Nunes, L. (2019). Hotel booking demand datasets. *Data in Brief, 22*, 41–49.
