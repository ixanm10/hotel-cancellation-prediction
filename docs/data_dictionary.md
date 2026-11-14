# Data dictionary – hotel_bookings.csv

32 columns, 119,390 rows. One row = one booking.

| Column | Type | Description |
|--------|------|-------------|
| hotel | Nominal | City Hotel or Resort Hotel |
| is_canceled | Nominal (label) | 1 = cancelled, 0 = not cancelled |
| lead_time | Ratio | Days between booking and arrival |
| arrival_date_year | Interval | Year of arrival |
| arrival_date_month | Ordinal | Month of arrival (text) |
| arrival_date_week_number | Ordinal | Week number of arrival |
| arrival_date_day_of_month | Ordinal | Day of arrival |
| stays_in_weekend_nights | Ratio | Saturday/Sunday nights booked |
| stays_in_week_nights | Ratio | Monday–Friday nights booked |
| adults | Ratio | Number of adults |
| children | Ratio | Number of children |
| babies | Ratio | Number of babies |
| meal | Nominal | BB (bed & breakfast), HB (half board), FB (full board), SC/Undefined (no meal) |
| country | Nominal | Country of origin (ISO 3166-1 alpha-3) |
| market_segment | Nominal | TA = travel agents, TO = tour operators, Direct, Groups, Corporate, ... |
| distribution_channel | Nominal | Booking channel |
| is_repeated_guest | Nominal | 1 = guest has stayed before |
| previous_cancellations | Ratio | Earlier bookings cancelled by the guest |
| previous_bookings_not_canceled | Ratio | Earlier bookings not cancelled |
| reserved_room_type | Nominal | Room type booked (code) |
| assigned_room_type | Nominal | Room type given at check-in (**leakage, not used**) |
| booking_changes | Ratio | Changes made to the booking |
| deposit_type | Nominal | No Deposit, Non Refund, Refundable |
| agent | Nominal | ID of the travel agency (missing = no agent) |
| company | Nominal | ID of the company (94% missing, **dropped**) |
| days_in_waiting_list | Ratio | Days before the booking was confirmed |
| customer_type | Nominal | Transient, Transient-Party, Contract, Group |
| adr | Ratio | Average daily rate (€) |
| required_car_parking_spaces | Ratio | Parking spaces requested |
| total_of_special_requests | Ratio | Special requests (e.g. twin bed, high floor) |
| reservation_status | Nominal | Check-Out, Canceled, No-Show (**leakage, not used**) |
| reservation_status_date | Interval | Date of the last status (**leakage, not used**) |

## Constructed attributes (src/pipeline.py)

| Column | Formula |
|--------|---------|
| total_nights | weekend + week nights |
| total_guests | adults + children + babies |
| has_kids | 1 if children + babies > 0 |
| total_revenue | adr × total_nights |
| weekend_ratio | weekend nights / total_nights |
| prev_cancel_rate | previous_cancellations / all previous bookings |
| arrival_dow | day of week of arrival (0 = Monday) |
| is_summer | 1 if arrival in June–August |
| lead_time_band | lead time grouped into 6 bands |
| country_grp | top 15 countries, others = "Other" |
