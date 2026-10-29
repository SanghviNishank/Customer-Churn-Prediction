# Executive Churn Analytics Report

## 1. Executive Summary & Core Metrics
- **Total Customer Base Analyzed:** 7,032
- **Total Churned Customers:** 1,869 (26.6%)
- **Retained Customers:** 5,163 (73.4%)
- **Monthly Revenue Associated with Churned Customers:** $139,130.85 (30.5% of total monthly billing)
- **Average Customer Tenure:** 32.4 months
- **Average Monthly Charge:** $64.80

---

## 2. Key Statistical Insights (Extracted from Dataset & Notebook)
1. **Contract Type is the Dominant Factor:**
   - Month-to-month contracts have an alarming **42.7% churn rate**.
   - One-year contracts show **11.3% churn**.
   - Two-year contracts demonstrate high loyalty with only **2.8% churn**.
   - *Conclusion:* Month-to-month customers are over 15x more likely to leave than two-year contract holders.

2. **Early Tenure Vulnerability:**
   - New customers (tenure 0-12 months) experience the highest churn rate (~47.4%).
   - Customers who stay past 24 months drop to under 20% churn, and past 48 months under 8%.

3. **Fiber Optic Service Quality & Pricing Friction:**
   - Fiber optic subscribers exhibit a **41.9% churn rate**, significantly higher than DSL (~19.0%) or no-internet users (~7.4%).
   - These users also bear higher monthly charges (average ~$91/month).

4. **Payment Method Association:**
   - Electronic check payers churn at **45.3%**, while automated bank transfers and credit card autopay users churn at ~15-16%.

---

## 3. Machine Learning Model Architecture & Performance
- **Selected Model:** Tuned XGBoost (`XGBClassifier`) with Stratified 80/20 train/test split.
- **Evaluation Criteria:** Prioritizing Recall and ROC-AUC on hold-out test set (1,407 customers).
- **Hold-out Test ROC-AUC:** 0.8380
- **Operational Decision Threshold:** 0.40 (Tuned from default 0.50)
  - At threshold 0.40, Recall increases from 51.1% to **66.3%**, successfully intercepting two-thirds of all at-risk customers.
  - Overall Accuracy: 79.0%
  - Test Precision: 59.3%

---

## 4. Prescriptive Business Recommendations
1. **'Lock & Save' Annual Conversion Program:**
   - Proactively target month-to-month customers in month 3 with an annual contract offer (e.g., 10% discount or 1 month free).
2. **First-Year Onboarding Care:**
   - Implement scheduled touchpoints during days 30, 60, and 90 to identify onboarding dissatisfaction early.
3. **Fiber Optic Quality & Value Audit:**
   - Conduct technical speed and stability checks on fiber lines and offer complimentary speed upgrades or streaming bundle discounts.
4. **Auto-Pay Migration Incentive:**
   - Offer a one-time $10 credit to electronic-check customers who migrate to automated bank debit or credit card payments.

*Generated automatically by Customer Churn Analytics Pipeline.*
