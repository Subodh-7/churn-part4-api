"""Tests for D2C Churn Scoring API — Subodh Raut"""
import sys, os; sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..'))
from fastapi.testclient import TestClient
from app.main import app
client = TestClient(app)

S={"customer_id":"CUST00001","city_tier":"Tier 1","age_group":"25-34",
    "acquisition_channel":"Instagram","loyalty_tier":"Silver",
    "preferred_category":"Skin Care","marketing_consent":"Yes",
    "recency_days":45.0,"frequency_180d":3.0,"monetary_180d":2500.0,
    "return_rate_180d":0.1,"avg_discount_pct_180d":0.2,"avg_rating_180d":4.0,
    "category_diversity_180d":2.0,"ticket_count_90d":1.0,"negative_ticket_rate_90d":0.0,
    "avg_resolution_hours_90d":5.0,"days_since_signup":300.0,"sessions_30d":5.0,
    "product_views_30d":12.0,"cart_adds_30d":2.0,"wishlist_adds_30d":1.0,
    "abandoned_carts_30d":1.0,"email_opens_30d":3.0,"campaign_clicks_30d":1.0,"last_visit_days_ago":5.0}
HR={**S,"customer_id":"CUST99999","recency_days":250.0,"sessions_30d":0.0,
    "frequency_180d":0.0,"monetary_180d":0.0,"last_visit_days_ago":100.0,
    "loyalty_tier":"None","ticket_count_90d":4.0,"negative_ticket_rate_90d":0.9}
LR={**S,"customer_id":"CUST88888","recency_days":2.0,"sessions_30d":25.0,
    "frequency_180d":8.0,"monetary_180d":9000.0,"last_visit_days_ago":0.0,"loyalty_tier":"Platinum"}

def test_health():
    r=client.get("/health"); assert r.status_code==200
    d=r.json(); assert d["status"]=="ok"; assert "test_auc_roc" in d
    print(f"\nHealth OK — AUC={d['test_auc_roc']}, threshold={d['threshold']}")

def test_predict_structure():
    r=client.post("/predict",json=S); assert r.status_code==200
    d=r.json()
    for k in ["customer_id","churn_probability","churn_predicted","risk_label","explanation"]:
        assert k in d
    assert d["customer_id"]=="CUST00001"
    assert 0<=d["churn_probability"]<=1
    assert d["churn_predicted"] in [0,1]
    assert d["risk_label"] in ["low_risk","medium_risk","high_risk"]
    print(f"\nPredict OK — prob={d['churn_probability']}, risk={d['risk_label']}")

def test_high_risk():
    r=client.post("/predict",json=HR); assert r.status_code==200
    d=r.json(); assert d["churn_predicted"]==1
    print(f"\nHigh-risk OK — prob={d['churn_probability']}")

def test_batch():
    r=client.post("/batch_predict",json={"customers":[S,HR,LR]}); assert r.status_code==200
    d=r.json(); assert d["total"]==3; assert len(d["predictions"])==3
    print(f"\nBatch OK — {d['total']} scored")

def test_empty_batch_400():
    r=client.post("/batch_predict",json={"customers":[]}); assert r.status_code==400

def test_missing_field_422():
    bad={k:v for k,v in S.items() if k!="recency_days"}
    r=client.post("/predict",json=bad); assert r.status_code==422

def test_prob_range():
    for c in [S,HR,LR]:
        r=client.post("/predict",json=c); assert 0<=r.json()["churn_probability"]<=1
    print("\nProb range OK")
