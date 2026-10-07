"""Input checks, lending-policy rules and the final decision.

Everything in this file is plain Python. The AI never changes a number or a
decision made here.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import pandas as pd

from model import emi, principal_for_emi

EMPLOYMENT = ["Salaried", "Self-employed", "Gig / contract"]
PURPOSES = ["Home renovation", "Medical", "Education", "Wedding", "Debt consolidation", "Travel", "Business working capital", "Other"]

# Product settings for the demo (unsecured personal loan)
MIN_LOAN, MAX_LOAN = 50_000, 25_00_000
TENURES = [12, 24, 36, 48, 60]
FOIR_CAP = 0.50  # used for the maximum eligible amount
FOIR_REFER, FOIR_REJECT = 0.50, 0.65

GRADES = [  # (upper PD bound, grade, model view)
    (0.04, "A", "Approve"),
    (0.08, "B", "Approve"),
    (0.15, "C", "Refer"),
    (0.25, "D", "Reject"),
    (1.01, "E", "Reject"),
]
SEVERITY = {"Approve": 0, "Refer": 1, "Reject": 2}

POLICY_TEXT = [
    ("P1", "Age 21 to 60, and loan must end before age 60", "Reject"),
    ("P2", "Net monthly income at least Rs 20,000", "Reject"),
    ("P3", "Credit score below 650", "Reject"),
    ("P4", "Credit score 650 to 699", "Refer"),
    ("P5", "Any 90+ days past due in last 24 months", "Reject"),
    ("P6", "EMI burden (FOIR) after this loan above 65%", "Reject"),
    ("P7", "EMI burden (FOIR) after this loan 50% to 65%", "Refer"),
    ("P8", "No credit history (new to credit)", "Refer"),
    ("P9", "More than 6 credit enquiries in 6 months", "Refer"),
    ("P10", "Less than 1 year in job (salaried) or 2 years in business", "Refer"),
]


def inr(x: float) -> str:
    """Indian digit grouping, e.g. 1234567 -> Rs 12,34,567."""
    neg = x < 0
    s = f"{abs(int(round(x))):d}"
    if len(s) > 3:
        head, tail = s[:-3], s[-3:]
        parts = []
        while len(head) > 2:
            parts.insert(0, head[-2:])
            head = head[:-2]
        if head:
            parts.insert(0, head)
        s = ",".join(parts) + "," + tail
    return ("-" if neg else "") + "Rs " + s


# ---------------------------------------------------------------- validation
def validate(a: dict) -> list[str]:
    """Return every problem at once, so the user can fix them in one go."""
    errs = []

    def num(key, label, lo, hi, allow_none=False):
        v = a.get(key)
        if v is None or (isinstance(v, float) and math.isnan(v)):
            if not allow_none:
                errs.append(f"{label} is missing.")
            return
        try:
            v = float(v)
        except (TypeError, ValueError):
            errs.append(f"{label} must be a number (got '{a.get(key)}').")
            return
        if not lo <= v <= hi:
            errs.append(f"{label} must be between {lo:,} and {hi:,} (got {v:,.0f}).")

    num("age", "Age", 18, 75)
    num("monthly_income", "Net monthly income", 1, 1_00_00_000)
    num("existing_emi", "Existing EMIs", 0, 1_00_00_000)
    num("loan_amount", "Loan amount", MIN_LOAN, MAX_LOAN)
    num("years_in_job", "Years in job/business", 0, 50)
    num("credit_score", "Credit score", 300, 900, allow_none=True)
    num("enquiries_6m", "Credit enquiries (6 months)", 0, 50)
    num("dpd30_24m", "Late payments 30+ days", 0, 24)
    num("card_utilisation", "Card utilisation", 0, 1.5)
    if a.get("tenure_months") not in TENURES:
        errs.append(f"Tenure must be one of {TENURES} months.")
    if a.get("employment") not in EMPLOYMENT:
        errs.append("Choose an employment type.")
    try:
        if float(a.get("existing_emi", 0)) >= float(a.get("monthly_income", 1)) > 0:
            errs.append("Existing EMIs are equal to or above monthly income. Please re-check both figures.")
    except (TypeError, ValueError):
        pass
    try:
        if float(a.get("years_in_job", 0)) > float(a.get("age", 99)) - 14:
            errs.append("Years in job is not possible for this age. Please re-check.")
    except (TypeError, ValueError):
        pass
    return errs


# ---------------------------------------------------------------- policy
@dataclass
class Assessment:
    pd: float
    grade: str
    model_view: str
    policy_hits: list = field(default_factory=list)  # (code, text, outcome)
    decision: str = "Refer"
    new_emi: float = 0.0
    foir_before: float = 0.0
    foir_after: float = 0.0
    max_eligible: float = 0.0
    counter_offer: float | None = None
    decided_by: str = ""


def grade_for(pd_: float):
    for ub, g, view in GRADES:
        if pd_ < ub:
            return g, view
    return "E", "Reject"


def assess(a: dict, pd_: float, rate: float) -> Assessment:
    g, view = grade_for(pd_)
    inc, ex = float(a["monthly_income"]), float(a["existing_emi"])
    new = emi(float(a["loan_amount"]), rate, int(a["tenure_months"]))
    fb, fa = ex / inc, (ex + new) / inc
    room = FOIR_CAP * inc - ex
    max_amt = max(0.0, principal_for_emi(room, rate, int(a["tenure_months"])))
    max_amt = min(MAX_LOAN, math.floor(max_amt / 10_000) * 10_000)
    hits = []
    age, tenure_yrs = float(a["age"]), int(a["tenure_months"]) / 12
    if age < 21 or age > 60 or age + tenure_yrs > 60:
        hits.append(("P1", f"Age {age:.0f} with a {int(a['tenure_months'])}-month loan is outside 21-60 at maturity", "Reject"))
    if inc < 20_000:
        hits.append(("P2", f"Income {inr(inc)} is below the Rs 20,000 floor", "Reject"))
    cs = a.get("credit_score")
    ntc = cs is None or (isinstance(cs, float) and math.isnan(cs))
    if not ntc and cs < 650:
        hits.append(("P3", f"Credit score {cs:.0f} is below 650", "Reject"))
    elif not ntc and cs < 700:
        hits.append(("P4", f"Credit score {cs:.0f} is in the 650-699 band", "Refer"))
    if int(a.get("dpd90_24m", 0)):
        hits.append(("P5", "A 90+ days past due event in the last 24 months", "Reject"))
    if fa > FOIR_REJECT:
        hits.append(("P6", f"FOIR after this loan is {fa:.0%}, above 65%", "Reject"))
    elif fa > FOIR_REFER:
        hits.append(("P7", f"FOIR after this loan is {fa:.0%}, between 50% and 65%", "Refer"))
    if ntc:
        hits.append(("P8", "No credit history, so the bureau score is missing", "Refer"))
    if float(a["enquiries_6m"]) > 6:
        hits.append(("P9", f"{int(a['enquiries_6m'])} credit enquiries in 6 months", "Refer"))
    need = 2 if a["employment"] == "Self-employed" else 1
    if float(a["years_in_job"]) < need:
        hits.append(("P10", f"{float(a['years_in_job']):g} years in current {'business' if need == 2 else 'job'} (needs {need})", "Refer"))

    worst_rule = max([SEVERITY[h[2]] for h in hits], default=0)
    final_sev = max(worst_rule, SEVERITY[view])
    decision = [k for k, v in SEVERITY.items() if v == final_sev][0]
    if final_sev == 0:
        by = "Model and policy agree"
    elif worst_rule == SEVERITY[view]:
        by = "Policy rules and risk model"
    elif worst_rule > SEVERITY[view]:
        by = "Policy rule" + ("s" if len([h for h in hits if SEVERITY[h[2]] == worst_rule]) > 1 else "")
    else:
        by = "Risk model"
    counter = None
    if decision != "Approve" and float(a["loan_amount"]) > max_amt >= MIN_LOAN and not any(h[2] == "Reject" and h[0] != "P6" for h in hits):
        counter = max_amt
    return Assessment(pd_, g, view, hits, decision, new, fb, fa, max_amt, counter, by)


def facts_for_ai(a: dict, s: Assessment, drivers: pd.DataFrame, rate: float) -> dict:
    """The only data the AI sees. No name, no free-text identity fields."""
    top = drivers.head(5)
    cs = a.get("credit_score")
    return {
        "decision": s.decision,
        "decided_by": s.decided_by,
        "risk_grade": s.grade,
        "estimated_default_probability_pct": round(s.pd * 100, 1),
        "applicant": {
            "age": int(a["age"]),
            "employment": a["employment"],
            "years_in_job": float(a["years_in_job"]),
            "net_monthly_income_rs": int(a["monthly_income"]),
            "existing_emis_rs": int(a["existing_emi"]),
            "credit_score": None if cs is None or (isinstance(cs, float) and math.isnan(cs)) else int(cs),
            "credit_enquiries_6m": int(a["enquiries_6m"]),
            "late_payments_30plus_24m": int(a["dpd30_24m"]),
            "card_utilisation_pct": round(float(a["card_utilisation"]) * 100),
            "loan_purpose": a.get("purpose", "Other"),
        },
        "loan": {
            "amount_rs": int(a["loan_amount"]),
            "tenure_months": int(a["tenure_months"]),
            "interest_rate_pct": rate,
            "new_emi_rs": round(s.new_emi),
            "foir_before_pct": round(s.foir_before * 100),
            "foir_after_pct": round(s.foir_after * 100),
            "max_eligible_amount_rs": int(s.max_eligible),
            "counter_offer_rs": None if s.counter_offer is None else int(s.counter_offer),
        },
        "policy_rules_triggered": [{"code": c, "rule": t, "outcome": o} for c, t, o in s.policy_hits],
        "top_model_drivers": [
            {
                "factor": r.factor,
                "applicant_value": r.shown_value,
                "portfolio_average": r.shown_avg,
                "effect": r.direction,
            }
            for r in top.itertuples()
        ],
    }


def expert_check(a: dict, s) -> tuple[str, str]:
    """A banker's rule of thumb, kept separate from the model on purpose."""
    cs = a.get("credit_score")
    clean = cs is not None and cs >= 750 and s.foir_after <= 0.45 and a["dpd30_24m"] == 0 and a["years_in_job"] >= 2
    weak = (cs is not None and cs < 650) or s.foir_after > 0.65 or a.get("dpd90_24m", 0)
    expect = "Approve" if clean else ("Reject" if weak else "Refer")
    return expect, ("agrees" if expect == s.decision else "differs")
