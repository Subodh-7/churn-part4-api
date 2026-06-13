"""D2C Churn Scoring API — Part 4 | Masai Capstone | Subodh Raut"""
import pickle, numpy as np
from pathlib import Path
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = Path(__file__).resolve().parent.parent / "model.pkl"
with open(MODEL_PATH,"rb") as f: payload=pickle.load(f)
MODEL=payload["model"]; FEAT=payload["feature_cols"]
THRESH=payload["threshold"]; METRICS=payload["metrics"]; LE=payload.get("le_dict",{})
print(f"Loaded: {len(FEAT)} features, threshold={THRESH:.3f}")

app = FastAPI(title="D2C Churn Scoring API",version="1.0.0",
    description="Predicts customer churn probability")

NUM=['recency_days','frequency_180d','monetary_180d','return_rate_180d','avg_discount_pct_180d',
    'avg_rating_180d','category_diversity_180d','ticket_count_90d','negative_ticket_rate_90d',
    'avg_resolution_hours_90d','days_since_signup','sessions_30d','product_views_30d','cart_adds_30d',
    'wishlist_adds_30d','abandoned_carts_30d','email_opens_30d','campaign_clicks_30d','last_visit_days_ago']
CAT=['city_tier','age_group','acquisition_channel','loyalty_tier','preferred_category','marketing_consent']

class Customer(BaseModel):
    customer_id:str=Field(...,example="CUST00001")
    city_tier:str=Field(...,example="Tier 1")
    age_group:str=Field(...,example="25-34")
    acquisition_channel:str=Field(...,example="Instagram")
    loyalty_tier:str=Field(...,example="Silver")
    preferred_category:str=Field(...,example="Skin Care")
    marketing_consent:str=Field(...,example="Yes")
    recency_days:float=Field(...,ge=0,example=45.0)
    frequency_180d:float=Field(...,ge=0,example=3.0)
    monetary_180d:float=Field(...,ge=0,example=2500.0)
    return_rate_180d:float=Field(...,ge=0,le=1,example=0.1)
    avg_discount_pct_180d:float=Field(...,ge=0,le=1,example=0.2)
    avg_rating_180d:float=Field(...,ge=1,le=5,example=4.0)
    category_diversity_180d:float=Field(...,ge=0,example=2.0)
    ticket_count_90d:float=Field(...,ge=0,example=1.0)
    negative_ticket_rate_90d:float=Field(...,ge=0,le=1,example=0.0)
    avg_resolution_hours_90d:float=Field(...,ge=0,example=5.0)
    days_since_signup:float=Field(...,ge=0,example=300.0)
    sessions_30d:float=Field(...,ge=0,example=5.0)
    product_views_30d:float=Field(...,ge=0,example=12.0)
    cart_adds_30d:float=Field(...,ge=0,example=2.0)
    wishlist_adds_30d:float=Field(...,ge=0,example=1.0)
    abandoned_carts_30d:float=Field(...,ge=0,example=1.0)
    email_opens_30d:float=Field(...,ge=0,example=3.0)
    campaign_clicks_30d:float=Field(...,ge=0,example=1.0)
    last_visit_days_ago:float=Field(...,ge=0,example=5.0)

class Prediction(BaseModel):
    customer_id:str; churn_probability:float; churn_predicted:int; risk_label:str; explanation:str

class BatchReq(BaseModel): customers:List[Customer]
class BatchRes(BaseModel): total:int; predictions:List[Prediction]

def enc(c):
    n=[getattr(c,col) for col in NUM]
    cat=[]
    for col in CAT:
        v=getattr(c,col)
        try: cat.append(float(LE[col].transform([str(v)])[0]) if col in LE else 0.0)
        except: cat.append(0.0)
    return np.array([n+cat])

def risk(p): return "high_risk" if p>=0.65 else ("medium_risk" if p>=THRESH else "low_risk")

def explain(c,p):
    r=[]
    if c.recency_days>90: r.append(f"no purchase in {int(c.recency_days)} days")
    if c.sessions_30d==0: r.append("zero sessions in last 30 days")
    if c.last_visit_days_ago>20: r.append(f"last visit {int(c.last_visit_days_ago)} days ago")
    if c.ticket_count_90d>=2: r.append(f"{int(c.ticket_count_90d)} tickets in 90d")
    if c.return_rate_180d>0.3: r.append(f"return rate {c.return_rate_180d:.0%}")
    if c.loyalty_tier=="None": r.append("not enrolled in loyalty programme")
    if not r: r.append("healthy engagement" if p<THRESH else "combination of weak signals")
    return "Risk drivers: "+"; ".join(r)+"."

@app.get("/health",tags=["Health"])
def health():
    return {"status":"ok","model":"LightGBM Churn Classifier v1.0","n_features":len(FEAT),
        "threshold":THRESH,"test_auc_roc":METRICS.get("test_auc_roc"),
        "test_f1":METRICS.get("test_f1"),"dataset":METRICS.get("dataset")}

@app.post("/predict",response_model=Prediction,tags=["Prediction"])
def predict(c:Customer):
    try:
        p=float(MODEL.predict_proba(enc(c))[0,1])
        return Prediction(customer_id=c.customer_id,churn_probability=round(p,4),
            churn_predicted=int(p>=THRESH),risk_label=risk(p),explanation=explain(c,p))
    except Exception as e: raise HTTPException(500,str(e))

@app.post("/batch_predict",response_model=BatchRes,tags=["Prediction"])
def batch(req:BatchReq):
    if len(req.customers)>1000: raise HTTPException(400,"Max 1000")
    if len(req.customers)==0: raise HTTPException(400,"Min 1 customer")
    try:
        X=np.vstack([enc(c) for c in req.customers])
        probs=MODEL.predict_proba(X)[:,1]
        return BatchRes(total=len(req.customers),predictions=[
            Prediction(customer_id=c.customer_id,churn_probability=round(float(p),4),
                churn_predicted=int(float(p)>=THRESH),risk_label=risk(float(p)),explanation=explain(c,float(p)))
            for c,p in zip(req.customers,probs)])
    except Exception as e: raise HTTPException(500,str(e))
