# Monitoring Plan — D2C Churn Scoring API
**Student:** Subodh Raut

## 1. Model Performance (Monthly)
| Metric | Baseline | Alert | Action |
|--------|----------|-------|--------|
| AUC-ROC | 0.8739 | <0.80 | Retrain |
| F1 | 0.8056 | <0.75 | Retrain |
| Predicted churn rate | ~47% | drift ±15% | Recalibrate threshold |
| Recall | 0.8631 | <0.78 | Lower threshold |

## 2. API Health (Real-time)
| Metric | Target | Alert |
|--------|--------|-------|
| Response time (p95) | <200ms | >500ms |
| Error rate | <1% | >5% |
| Uptime | 99.9% | <99% |

## 3. Data Drift (Monthly)
Track: `recency_days` mean/std, `sessions_30d` mean/% zero, `monetary_180d` mean, `ticket_count_90d` mean, `last_visit_days_ago` mean.
**Trigger:** any feature shifts >2 std dev from training baseline → model review.

## 4. Business Outcome Tracking
1. Weekly: export high-risk predictions (churn_predicted=1)
2. After 60 days: check actual churn for that cohort
3. Compute actual precision/recall, track campaign conversion by risk segment

## 5. Retraining Schedule
| Trigger | Action |
|---------|--------|
| Monthly | Check metrics on last 30 days |
| AUC < 0.80 | Immediate retrain |
| Major pricing/product change | Force retrain |
| Scheduled | Full retrain every 3 months |

## 6. Responsible Use
### ✅ Appropriate
- Personalised win-back campaigns for high-risk customers
- Prioritising retention team calls
- Adjusting loyalty outreach frequency

### ❌ Inappropriate
- Denying service/credit based solely on score
- Sharing scores externally
- Using scores >30 days old without rescoring
- Automating irreversible actions on model output alone

### ⚠️ Human Review Required
- prob > 0.90 AND monetary_180d > ₹5,000 (high-value at risk)
- Any campaign with discount > 25%
- Complaint escalations triggered by model score
