# 🏨 Hotel Booking Cancellation Prediction

**CDS 4003 – Data Science for Business Application | Group 3**

A machine learning model that predicts at booking time whether a hotel reservation will be cancelled, so staff can act before the room is lost. It's built on 119,390 bookings from a City Hotel (Lisbon) and a Resort Hotel (Algarve), 2015–2017.

**🚀 Live app:** https://hotel-cancellation-group3.streamlit.app

**📓 Notebook (Google Colab):** https://colab.research.google.com/drive/1ECBCZcbrHZaAUsSFODAoYim1JToiv2kH?usp=sharing

![ML Pipeline](https://github.com/ixanm10/hotel-cancellation-prediction/actions/workflows/pipeline.yml/badge.svg)

## Team

| Name | GitHub | Role |
|------|--------|------|
| Amna Nasser Alshamsi | [Ixanm10](https://github.com/ixanm10) | Model evaluation and comparison, Streamlit app, explaining results for hotel staff |
| Amna Ali Alhosani | [amna2116](https://github.com/amna2116) | Data understanding and cleaning: missing values, duplicates, inconsistent categories, outliers, data quality checks |
| Fatema Ahmed Alyileili | [FatmaAhmedAlyileili](https://github.com/FatmaAhmedAlyileili) | Business problem and objectives, dataset comparison and justification (source, licence, limitations), Python code support |

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
├── docs/dataset_search.md           # datasets considered, accepted and rejected
├── notebooks/01_stages_1-3_business_data_cleaning.ipynb   # Stages 1–3 only
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

**Notebooks in Google Colab:**
- Stages 1–3 (business problem, data collection, cleaning): [Open in Colab](https://colab.research.google.com/github/ixanm10/hotel-cancellation-prediction/blob/main/notebooks/01_stages_1-3_business_data_cleaning.ipynb) – the data loads from this repository automatically.
- Full analysis: [Open in Colab](https://colab.research.google.com/github/ixanm10/hotel-cancellation-prediction/blob/main/notebooks/Hotel_Cancellation_Colab.ipynb) – the data loads from this repository automatically (about 5 minutes).

## Dataset

- **Name:** Hotel Booking Demand (`data/hotel_bookings.csv`), 119,390 bookings × 32 columns, July 2015 – August 2017
- **Download:** https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand (a copy is already in `data/`)
- **Original source:** Antonio, N., de Almeida, A., & Nunes, L. (2019). Hotel booking demand datasets. *Data in Brief, 22*, 41–49. https://doi.org/10.1016/j.dib.2018.11.126
- **Licence:** CC BY 4.0 – academic use allowed with attribution
- **Size:** below the 200,000-row guideline; approved by the instructor at Milestone 2
- Dataset search evidence: [`docs/dataset_search.md`](docs/dataset_search.md) · column descriptions: [`docs/data_dictionary.md`](docs/data_dictionary.md)


