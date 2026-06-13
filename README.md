# Part 4 — FastAPI Churn Scoring Service
**Student:** Subodh Raut | **Dataset:** Masai D2C Churn Capstone Real Data

## Structure
```
part4_api/
├── app/main.py       # FastAPI app
├── tests/test_api.py # 7 tests
├── model.pkl
├── monitoring_plan.md
└── requirements.txt
```

## How to Run
```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```
Open: http://localhost:8000/docs

## Run Tests
```bash
pytest tests/test_api.py -v
```
All 7 tests pass.

## Endpoints

### GET /health
```bash
curl http://localhost:8000/health
```
```json
{"status":"ok","model":"LightGBM Churn Classifier v1.0","threshold":0.441,"test_auc_roc":0.8739,"test_f1":0.8056}
```

### POST /predict
```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{
  "customer_id":"CUST00001","city_tier":"Tier 1","age_group":"25-34",
  "acquisition_channel":"Instagram","loyalty_tier":"Silver",
  "preferred_category":"Skin Care","marketing_consent":"Yes",
  "recency_days":45.0,"frequency_180d":3.0,"monetary_180d":2500.0,
  "return_rate_180d":0.1,"avg_discount_pct_180d":0.2,"avg_rating_180d":4.0,
  "category_diversity_180d":2.0,"ticket_count_90d":1.0,"negative_ticket_rate_90d":0.0,
  "avg_resolution_hours_90d":5.0,"days_since_signup":300.0,"sessions_30d":5.0,
  "product_views_30d":12.0,"cart_adds_30d":2.0,"wishlist_adds_30d":1.0,
  "abandoned_carts_30d":1.0,"email_opens_30d":3.0,"campaign_clicks_30d":1.0,"last_visit_days_ago":5.0
}'
```
Response:
```json
{"customer_id":"CUST00001","churn_probability":0.23,"churn_predicted":0,"risk_label":"low_risk","explanation":"Risk drivers: healthy engagement."}
```

### POST /batch_predict
Send `{"customers":[...]}` — up to 1000 customers.

## Risk Labels
| Label | Probability |
|-------|-------------|
| high_risk | ≥0.65 |
| medium_risk | 0.441–0.65 |
| low_risk | <0.441 |
