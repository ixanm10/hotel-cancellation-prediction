# Dataset search evidence (Stage 2 / Milestone 2)

| Source | Search terms | Dataset | Decision | Reason |
|---|---|---|---|---|
| Kaggle | hotel booking cancellation | Hotel Booking Demand (Mostipak / Antonio et al.) | **Accepted** | Real booking records, 119,390 rows × 32 columns, clear binary target and strong business relevance |
| Kaggle | hotel reservations dataset | Hotel Reservations Classification Dataset | Rejected | Only about 36,000 rows and 19 columns, below the size requirement |
| Bayanat (UAE Open Data) | hotel establishments Dubai | Hotel establishment statistics | Rejected | Aggregated monthly totals, not individual bookings, so it cannot support a booking-level model |

**Size requirement:** the brief asks for 200,000+ rows. The selected dataset has 119,390 rows and 32 columns; the instructor approved it at Milestone 2.

## Selected dataset

- **Name:** Hotel Booking Demand (`hotel_bookings.csv`)
- **Kaggle:** https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand
- **Original source:** Antonio, N., de Almeida, A. and Nunes, L. (2019). Hotel booking demand datasets. *Data in Brief*, 22, 41–49. https://doi.org/10.1016/j.dib.2018.11.126
- **Licence:** CC BY 4.0 – academic use allowed with attribution
- **Target:** `is_canceled` (1 = cancelled, 0 = not cancelled) – binary classification
