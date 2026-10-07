"""Risk model: logistic regression on synthetic data, with per-factor drivers.

Drivers are exact for a linear model: each feature's contribution to the
log-odds is coef * (standardised value - 0), so contributions plus the
intercept add up to the score. No black-box explainer is needed.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

DATA = Path(__file__).parent / "data" / "training_sample.csv.gz"
NTC_SCORE = 700  # value used for the model when an applicant has no bureau history

FEATURES = [
    "credit_score_f",
    "foir_after",
    "loan_to_income",
    "years_in_job",
    "emp_self",
    "emp_gig",
    "enquiries_6m",
    "dpd30_24m",
    "card_utilisation",
    "home_owner",
    "new_to_credit",
]

LABELS = {
    "credit_score_f": "Credit score",
    "foir_after": "EMI burden (FOIR)",
    "loan_to_income": "Loan size vs annual income",
    "years_in_job": "Years in current job/business",
    "emp_self": "Self-employed",
    "emp_gig": "Gig / contract work",
    "enquiries_6m": "Recent credit enquiries",
    "dpd30_24m": "Late payments (30+ days)",
    "card_utilisation": "Credit card utilisation",
    "home_owner": "Owns home",
    "new_to_credit": "No credit history",
}


BINARY = {"emp_self", "emp_gig", "home_owner", "new_to_credit"}


def fmt_value(f: str, v: float, avg: bool = False) -> str:
    if f in BINARY:
        return f"{v:.0%} of applicants" if avg else ("Yes" if v >= 0.5 else "No")
    if f in ("foir_after", "card_utilisation"):
        return f"{v:.0%}"
    if f == "loan_to_income":
        return f"{v:.2f}x annual income"
    if f == "years_in_job":
        return f"{v:.1f} yrs"
    if f == "credit_score_f":
        return f"{v:.0f}"
    return f"{v:.1f}" if avg else f"{v:.0f}"


def emi(principal: float, annual_rate: float, months: int) -> float:
    r = annual_rate / 1200
    if r == 0:
        return principal / months
    return principal * r * (1 + r) ** months / ((1 + r) ** months - 1)


def principal_for_emi(emi_amt: float, annual_rate: float, months: int) -> float:
    r = annual_rate / 1200
    if emi_amt <= 0:
        return 0.0
    if r == 0:
        return emi_amt * months
    return emi_amt * ((1 + r) ** months - 1) / (r * (1 + r) ** months)


def build_features(df: pd.DataFrame, rate: float) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    ntc = df["credit_score"].isna()
    out["credit_score_f"] = df["credit_score"].fillna(NTC_SCORE)
    new_emi = [emi(p, rate, int(t)) for p, t in zip(df["loan_amount"], df["tenure_months"])]
    out["foir_after"] = (df["existing_emi"] + np.array(new_emi)) / df["monthly_income"]
    out["loan_to_income"] = df["loan_amount"] / (df["monthly_income"] * 12)
    out["years_in_job"] = df["years_in_job"]
    out["emp_self"] = (df["employment"] == "Self-employed").astype(int)
    out["emp_gig"] = (df["employment"] == "Gig / contract").astype(int)
    out["enquiries_6m"] = df["enquiries_6m"]
    out["dpd30_24m"] = df["dpd30_24m"]
    out["card_utilisation"] = df["card_utilisation"]
    out["home_owner"] = df["home_owner"].astype(int)
    out["new_to_credit"] = ntc.astype(int)
    return out


@dataclass
class RiskModel:
    clf: LogisticRegression
    mean: pd.Series
    std: pd.Series
    auc: float
    n_train: int
    base_rate: float

    def _z(self, X: pd.DataFrame) -> pd.DataFrame:
        return (X[FEATURES] - self.mean) / self.std

    def predict_pd(self, X: pd.DataFrame) -> np.ndarray:
        return self.clf.predict_proba(self._z(X))[:, 1]

    def drivers(self, X_row: pd.DataFrame) -> pd.DataFrame:
        """Log-odds contribution of each feature for one applicant."""
        z = self._z(X_row).iloc[0]
        contrib = z * self.clf.coef_[0]
        d = pd.DataFrame(
            {
                "feature": FEATURES,
                "factor": [LABELS[f] for f in FEATURES],
                "value": [X_row.iloc[0][f] for f in FEATURES],
                "portfolio_avg": [self.mean[f] for f in FEATURES],
                "contribution": contrib.values,
            }
        )
        if X_row.iloc[0]["new_to_credit"] == 1:
            d.loc[d.feature == "credit_score_f", "factor"] = "Credit score (assumed 700, no history)"
        d["shown_value"] = [fmt_value(f, v) for f, v in zip(d.feature, d.value)]
        d["shown_avg"] = [fmt_value(f, v, True) for f, v in zip(d.feature, d.portfolio_avg)]
        d["direction"] = np.where(d["contribution"] > 0, "Raises risk", "Lowers risk")
        return d.reindex(d["contribution"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def train(rate: float = 13.0) -> RiskModel:
    if not DATA.exists():  # the synthetic file is rebuilt from a fixed seed if missing
        import make_data
        make_data.main()
    df = pd.read_csv(DATA)
    X = build_features(df, rate)
    y = df["defaulted_12m"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=7, stratify=y)
    mean, std = Xtr[FEATURES].mean(), Xtr[FEATURES].std().replace(0, 1)
    clf = LogisticRegression(max_iter=500, C=1.0)
    clf.fit((Xtr[FEATURES] - mean) / std, ytr)
    auc = roc_auc_score(yte, clf.predict_proba((Xte[FEATURES] - mean) / std)[:, 1])
    return RiskModel(clf, mean, std, float(auc), len(Xtr), float(y.mean()))
