"""Generate the synthetic data used by CreditLens.

Nothing here is real customer data. The training file is drawn from a
documented data-generating process so that the model's behaviour can be
explained and checked. Run once: python make_data.py
"""
import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(65015)
OUT = Path(__file__).parent / "data"
OUT.mkdir(exist_ok=True)


def _applicants(n: int) -> pd.DataFrame:
    emp = RNG.choice(["Salaried", "Self-employed", "Gig / contract"], size=n, p=[0.62, 0.25, 0.13])
    age = RNG.integers(21, 59, size=n)
    income = np.where(
        emp == "Salaried",
        RNG.lognormal(np.log(52000), 0.45, n),
        np.where(emp == "Self-employed", RNG.lognormal(np.log(60000), 0.6, n), RNG.lognormal(np.log(28000), 0.35, n)),
    ).round(-2)
    income = np.clip(income, 15000, 600000)
    ntc = RNG.random(n) < np.where(emp == "Gig / contract", 0.25, 0.07)
    score = np.clip(RNG.normal(735, 55, n), 520, 890).round()
    years = np.clip(RNG.gamma(2.2, 2.2, n), 0, 35).round(1)
    existing_emi = (income * np.clip(RNG.beta(1.6, 5, n), 0, 0.6)).round(-2)
    existing_emi[RNG.random(n) < 0.35] = 0
    loan = np.clip((income * RNG.uniform(2, 14, n)).round(-4), 50000, 2500000)
    tenure = RNG.choice([12, 24, 36, 48, 60], size=n, p=[0.1, 0.25, 0.3, 0.15, 0.2])
    enq = RNG.poisson(1.3, n)
    dpd30 = RNG.poisson(0.25, n)
    util = np.clip(RNG.beta(2, 4, n), 0, 1).round(2)
    owned = (RNG.random(n) < 0.38).astype(int)
    df = pd.DataFrame(
        dict(
            age=age,
            employment=emp,
            years_in_job=years,
            monthly_income=income,
            existing_emi=existing_emi,
            loan_amount=loan,
            tenure_months=tenure,
            credit_score=np.where(ntc, np.nan, score),
            enquiries_6m=enq,
            dpd30_24m=dpd30,
            card_utilisation=util,
            home_owner=owned,
        )
    )
    return df


def main():
    from model import build_features  # local import keeps this script light

    df = _applicants(6000)
    X = build_features(df, rate=13.0)
    # Documented data-generating process (true log-odds of default)
    logit = (
        -3.65
        - 0.011 * (X["credit_score_f"] - 740)
        + 3.2 * (X["foir_after"] - 0.40)
        + 0.22 * (X["loan_to_income"] - 0.9)
        - 0.07 * (X["years_in_job"].clip(upper=15) - 4)
        + 0.35 * X["emp_self"]
        + 0.75 * X["emp_gig"]
        + 0.17 * X["enquiries_6m"]
        + 0.65 * X["dpd30_24m"]
        + 1.3 * (X["card_utilisation"] - 0.3)
        - 0.30 * X["home_owner"]
        + 0.45 * X["new_to_credit"]
    )
    pd_true = 1 / (1 + np.exp(-logit))
    df["defaulted_12m"] = (RNG.random(len(df)) < pd_true).astype(int)
    df.to_csv(OUT / "training_sample.csv.gz", index=False)
    print("training rows", len(df), "default rate", round(df.defaulted_12m.mean(), 4))

    # 25-row batch file for the bulk screening demo
    b = _applicants(25).drop(columns=[])
    b.insert(0, "applicant_ref", [f"APP-{2001 + i}" for i in range(25)])
    b.loc[3, "monthly_income"] = 12000  # below income floor
    b.loc[7, "age"] = 19  # invalid age, should be caught
    b.loc[11, "credit_score"] = 1200  # out of range, should be caught
    b["dpd90_24m"] = 0
    b.loc[15, "dpd90_24m"] = 1  # serious delinquency, policy knock-out
    b.to_csv(OUT / "sample_batch.csv", index=False)
    print("batch rows", len(b))


if __name__ == "__main__":
    main()
