"""CreditLens: an AI-assisted loan eligibility and credit-risk pre-screener.

Run locally:  streamlit run app.py
"""
from __future__ import annotations

import hashlib
import io
import json
import math
from datetime import datetime

import pandas as pd
import streamlit as st

import ai
from model import FEATURES, build_features, emi, train
from rules import (expert_check, EMPLOYMENT, GRADES, MAX_LOAN, MIN_LOAN, POLICY_TEXT, PURPOSES, TENURES, assess, facts_for_ai,
                   inr, validate)

st.set_page_config(page_title="CreditLens · Loan pre-screener", page_icon="🔎", layout="wide")

st.markdown(
    """
<style>
.block-container {padding-top: 2.2rem; max-width: 1180px;}
h1 {font-weight: 750; letter-spacing: -0.02em;}
.decision {border-radius: 10px; padding: 14px 18px; margin: 6px 0 14px 0; font-size: 1.05rem;}
.decision b {font-size: 1.35rem;}
.Approve {background: #E3F4EC; border-left: 6px solid #1E8E5A; color: #0F4D30;}
.Refer {background: #FFF4DB; border-left: 6px solid #D99A00; color: #6B4A00;}
.Reject {background: #FCE4E4; border-left: 6px solid #C53030; color: #7A1A1A;}
.small {font-size: 0.82rem; color: #5B6B7B;}
.pill {display:inline-block; padding:2px 9px; border-radius: 99px; font-size: 0.78rem; margin-right: 6px;}
.ok {background:#E3F4EC; color:#0F4D30;} .warn {background:#FFF4DB; color:#6B4A00;} .bad {background:#FCE4E4; color:#7A1A1A;}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def get_model():
    return train(13.0)


MODEL = get_model()

SAMPLES = {
    "Kavya Sharma · salaried IT analyst": dict(
        name="Kavya Sharma", age=29, employment="Salaried", years_in_job=4.0, monthly_income=85000, existing_emi=9000,
        loan_amount=600000, tenure_months=48, credit_score=782, ntc=False, enquiries_6m=1, dpd30_24m=0, dpd90_24m=0,
        card_utilisation=0.22, home_owner=0, purpose="Home renovation", notes=""),
    "Arjun Mehta · delivery partner (gig)": dict(
        name="Arjun Mehta", age=26, employment="Gig / contract", years_in_job=1.5, monthly_income=32000, existing_emi=4000,
        loan_amount=150000, tenure_months=24, credit_score=None, ntc=True, enquiries_6m=2, dpd30_24m=0, dpd90_24m=0,
        card_utilisation=0.0, home_owner=0, purpose="Medical", notes="Need funds for my mother's surgery next month."),
    "Sunil Verma · kirana shop owner": dict(
        name="Sunil Verma", age=44, employment="Self-employed", years_in_job=9.0, monthly_income=55000, existing_emi=21000,
        loan_amount=900000, tenure_months=36, credit_score=688, ntc=False, enquiries_6m=5, dpd30_24m=2, dpd90_24m=0,
        card_utilisation=0.71, home_owner=1, purpose="Business working capital", notes=""),
    "Blank form": dict(
        name="", age=30, employment="Salaried", years_in_job=2.0, monthly_income=40000, existing_emi=0,
        loan_amount=200000, tenure_months=36, credit_score=720, ntc=False, enquiries_6m=0, dpd30_24m=0, dpd90_24m=0,
        card_utilisation=0.3, home_owner=0, purpose="Other", notes=""),
}

ss = st.session_state
ss.setdefault("history", [])
ss.setdefault("notes_cache", {})
ss.setdefault("qa", [])

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### 🔎 CreditLens")
    st.caption("AI-assisted loan pre-screener for personal loans")
    rate = st.slider("Product interest rate (% p.a.)", 10.5, 24.0, 13.0, 0.5,
                     help="Used to compute the EMI. The demo product is an unsecured personal loan.")
    outage = st.toggle("Simulate AI outage", value=False, help="Shows what the app does when Gemini is unavailable.")
    key, model_name, backup_name = ai.get_settings()
    if outage:
        st.warning("AI outage simulated. Notes use the rule-based template.")
    elif key:
        st.success(f"Gemini connected · `{model_name}`")
    else:
        st.warning("Gemini key not set. Notes use the rule-based template.")
    st.divider()
    st.markdown(
        f"<div class='small'><b>Risk model:</b> logistic regression, {MODEL.n_train:,} synthetic training rows, "
        f"holdout AUC {MODEL.auc:.2f}.<br><br><b>Decision maker:</b> lending-policy rules + risk model. "
        "Gemini only explains the result.<br><br>🔒 <b>Privacy:</b> the applicant's name is never sent to Gemini; "
        "only the computed figures and the optional notes are. Nothing is stored after you close the tab.<br><br>"
        "Pre-screening only. Not a credit decision or financial advice.</div>",
        unsafe_allow_html=True,
    )

st.title("CreditLens")
st.markdown("Screen a personal-loan applicant in under a minute: a risk score, the reasons behind it, and a clear "
            "**Approve / Refer / Reject** recommendation, explained in plain language.")

tab1, tab2, tab3, tab4 = st.tabs(["Single applicant", "Batch screening", "Session history", "How it works"])


# ---------------------------------------------------------------- helpers
def score_one(a: dict, rate_: float):
    df = pd.DataFrame([{**a, "credit_score": (float("nan") if a.get("credit_score") is None else a["credit_score"])}])
    X = build_features(df, rate_)
    pd_ = float(MODEL.predict_pd(X)[0])
    s = assess({**a, "credit_score": df["credit_score"].iloc[0]}, pd_, rate_)
    drivers = MODEL.drivers(X)
    return s, drivers


def a_hash(a: dict, rate_: float) -> str:
    return hashlib.sha1(json.dumps({**a, "rate": rate_}, sort_keys=True, default=str).encode()).hexdigest()[:10]


SEV = {"Approve": 0, "Refer": 1, "Reject": 2}
DECISION_ICON = {"Approve": "✅", "Refer": "🟡", "Reject": "⛔"}

# ---------------------------------------------------------------- tab 1
with tab1:
    st.subheader("① Applicant details")
    sample = st.radio("Start from", list(SAMPLES), horizontal=True, label_visibility="collapsed")
    d = SAMPLES[sample]
    k = sample[:5]
    with st.form("applicant"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Applicant**")
            name = st.text_input("Applicant name (stays in the app)", d["name"], key=k + "name")
            age = st.number_input("Age", 18, 75, d["age"], key=k + "age")
            employment = st.selectbox("Employment type", EMPLOYMENT, EMPLOYMENT.index(d["employment"]), key=k + "emp")
            years = st.number_input("Years in current job / business", 0.0, 50.0, float(d["years_in_job"]), 0.5, key=k + "yrs")
            home = st.checkbox("Owns home", bool(d["home_owner"]), key=k + "home")
        with c2:
            st.markdown("**Income and loan**")
            income = st.number_input("Net monthly income (Rs)", 0, 10_000_000, d["monthly_income"], 1000, key=k + "inc")
            existing = st.number_input("Existing EMIs per month (Rs)", 0, 10_000_000, d["existing_emi"], 500, key=k + "ex")
            loan = st.number_input(f"Loan amount (Rs {MIN_LOAN:,} to {MAX_LOAN:,})", 0, 50_000_000, d["loan_amount"], 10000, key=k + "loan")
            tenure = st.select_slider("Tenure (months)", TENURES, d["tenure_months"], key=k + "ten")
            purpose = st.selectbox("Purpose", PURPOSES, PURPOSES.index(d["purpose"]), key=k + "pur")
        with c3:
            st.markdown("**Credit profile**")
            ntc = st.checkbox("No credit history (new to credit)", d["ntc"], key=k + "ntc")
            score = st.number_input("Credit score (300-900)", 0, 2000, d["credit_score"] or 700, key=k + "cs", disabled=False)
            enq = st.number_input("Credit enquiries in last 6 months", 0, 50, d["enquiries_6m"], key=k + "enq")
            dpd30 = st.number_input("Late payments (30+ days) in 24 months", 0, 24, d["dpd30_24m"], key=k + "dpd")
            dpd90 = st.checkbox("Any 90+ days overdue in 24 months", bool(d["dpd90_24m"]), key=k + "d90")
            util = st.slider("Credit card utilisation", 0.0, 1.0, float(d["card_utilisation"]), 0.01, key=k + "util",
                             format="%.2f")
        notes = st.text_area("Applicant notes (optional, unverified)", d["notes"], max_chars=600, key=k + "notes",
                             placeholder="Anything the applicant wants the lender to know")
        submitted = st.form_submit_button("Screen application", type="primary")

    if submitted:
        a = dict(age=age, employment=employment, years_in_job=years, monthly_income=income, existing_emi=existing,
                 loan_amount=loan, tenure_months=tenure, credit_score=None if ntc else score, enquiries_6m=enq,
                 dpd30_24m=dpd30, dpd90_24m=int(dpd90), card_utilisation=util, home_owner=int(home), purpose=purpose)
        errs = validate(a)
        if errs:
            ss.pop("current", None)
            st.error("Please fix these before screening:\n\n" + "\n".join(f"- {e}" for e in errs))
        else:
            h = a_hash(a, rate)
            prev = next((x for x in ss.history if x["hash"] == h), None)
            if prev:
                st.info(f"This exact application was already screened at {prev['time']}. Showing the saved result; no new AI call.")
            else:
                s, _ = score_one(a, rate)
                ss.history.append(dict(hash=h, time=datetime.now().strftime("%H:%M:%S"), name=name or "(no name)",
                                       loan=inr(loan), decision=s.decision, grade=s.grade, pd=f"{s.pd:.1%}"))
            ss.current = dict(a=a, name=name, notes=notes, rate=rate, hash=h)
            ss.qa = []

    cur = ss.get("current")
    if cur and cur["rate"] != rate:
        cur["rate"] = rate
        cur["hash"] = a_hash(cur["a"], rate)
    if cur:
        a, rate_ = cur["a"], cur["rate"]
        s, drivers = score_one(a, rate_)
        facts = facts_for_ai(a, s, drivers, rate_)
        if s.counter_offer:  # only offer a smaller amount to the AI if it actually improves the result
            s_c, _ = score_one({**a, "loan_amount": s.counter_offer}, rate_)
            if SEV[s_c.decision] >= SEV[s.decision]:
                facts["loan"]["counter_offer_rs"] = None
                facts["loan"]["counter_offer_note"] = "A smaller loan within the 50% FOIR cap would not change the decision."

        st.divider()
        st.subheader("② Pre-screening result")
        st.markdown(
            f"<div class='decision {s.decision}'>{DECISION_ICON[s.decision]} <b>{s.decision.upper()}</b> &nbsp;·&nbsp; "
            f"{cur['name'] or 'Applicant'} &nbsp;·&nbsp; {inr(a['loan_amount'])} for {a['tenure_months']} months"
            f"<br><span style='font-size:0.9rem'>Decided by: {s.decided_by}. Risk model view: {s.model_view} (grade {s.grade}).</span></div>",
            unsafe_allow_html=True,
        )
        m = st.columns(3)
        m[0].metric("Risk grade", s.grade)
        m[1].metric("Default probability (model)", f"{s.pd:.1%}")
        m[2].metric("Policy rules triggered", len(s.policy_hits))
        m = st.columns(3)
        m[0].metric("New EMI", inr(s.new_emi))
        m[1].metric("EMI burden (FOIR) before → after", f"{s.foir_before:.0%} → {s.foir_after:.0%}")
        m[2].metric("Max eligible within 50% FOIR", inr(s.max_eligible))

        cL, cR = st.columns([1, 1])
        with cL:
            st.markdown("**What drove the risk score** (model, log-odds contribution)")
            dv = drivers.head(7)
            mx = max(dv["contribution"].abs().max(), 0.5)
            rows_html = ""
            for r in dv.itertuples():
                w = abs(r.contribution) / mx * 50
                col = "#C53030" if r.contribution > 0 else "#1E8E5A"
                left = 50 if r.contribution > 0 else 50 - w
                rows_html += (
                    f"<div class='drv' style='display:flex;align-items:center;margin:5px 0;font-size:0.85rem'>"
                    f"<div style='width:46%;text-align:right;padding-right:10px'>{r.factor}<br><span class='small'>{r.shown_value} · avg {r.shown_avg}</span></div>"
                    f"<div style='width:54%;position:relative;height:18px;background:linear-gradient(90deg,transparent 49.6%,#9AA5B1 49.6%,#9AA5B1 50.4%,transparent 50.4%)'>"
                    f"<div style='position:absolute;left:{left:.1f}%;width:{w:.1f}%;height:18px;background:{col};border-radius:3px'></div></div></div>")
            st.markdown(rows_html + "<div class='small' style='display:flex'><div style='width:46%'></div><div style='width:54%;display:flex;justify-content:space-between'>"
                        "<span style='color:#1E8E5A'>◀ lowers risk</span><span style='color:#C53030'>raises risk ▶</span></div></div>", unsafe_allow_html=True)
        with cR:
            st.markdown("**Lending-policy rules**")
            if s.policy_hits:
                st.dataframe(pd.DataFrame(s.policy_hits, columns=["Rule", "Finding", "Outcome"]), hide_index=True, width="stretch",
                             column_config={"Rule": st.column_config.TextColumn(width=45), "Outcome": st.column_config.TextColumn(width=70)})
            else:
                st.success("No policy rule triggered.")
            exp, verdict = expert_check(a, s)
            if verdict == "agrees":
                st.markdown(f"<span class='pill ok'>Rule-of-thumb check</span> A banker's quick rule also says <b>{exp}</b>.", unsafe_allow_html=True)
            else:
                st.markdown(f"<span class='pill warn'>Rule-of-thumb check</span> A banker's quick rule says <b>{exp}</b>, "
                            f"the app says <b>{s.decision}</b>. Review the drivers before acting.", unsafe_allow_html=True)
            st.caption("Rule of thumb: score ≥ 750, FOIR ≤ 45%, no late payments and 2+ years in job → Approve; "
                       "score < 650, FOIR > 65% or a 90+ DPD → Reject; otherwise Refer.")
            if s.counter_offer and s.decision != "Approve":
                s2, _ = score_one({**a, "loan_amount": s.counter_offer}, rate_)
                if SEV[s2.decision] < SEV[s.decision]:
                    st.info(f"Counter-offer: at {inr(s.counter_offer)} (within the 50% EMI cap) the result would improve to **{s2.decision}**.")
                else:
                    st.info(f"Even at {inr(s.counter_offer)} (within the 50% EMI cap) the result stays **{s2.decision}**, so a smaller loan alone does not fix this application.")

        # ---------------- AI explanation
        st.divider()
        st.subheader("③ Explanation")
        inj = ai.injection_phrases(cur["notes"])
        if inj:
            st.error(f"🚩 The applicant notes contain instruction-like text ({', '.join(inj)}). They are treated as data only "
                     "and cannot change the decision.")
        ck = cur["hash"] + ("-off" if outage else "")
        if ck not in ss.notes_cache:
            if outage:
                ss.notes_cache[ck] = (ai.fallback_note(facts), "rule-based template", "AI outage simulated.")
            else:
                with st.spinner("Gemini is writing the explanation…"):
                    try:
                        note, used = ai.write_note(facts, cur["notes"])
                        ss.notes_cache[ck] = (note, used, None)
                    except ai.AIUnavailable as e:
                        ss.notes_cache[ck] = (ai.fallback_note(facts), "rule-based template", str(e))
        note, used, reason = ss.notes_cache[ck]
        if reason:
            st.warning(f"AI explanation unavailable: {reason} Showing the rule-based note instead.")
        n1, n2 = st.columns(2)
        with n1:
            st.markdown("**For the loan officer**")
            st.write(note.officer_summary)
            for r in note.key_reasons:
                st.markdown(f"- **{r.factor}:** {r.explanation}")
            st.markdown("**Verify before sanction**")
            for c in note.checks_for_officer:
                st.markdown(f"- {c}")
        with n2:
            st.markdown("**Message for the applicant**")
            st.info(note.applicant_message)
            st.markdown("**How to improve**")
            for t in note.improvement_tips:
                st.markdown(f"- {t}")
        rv = ai.review(ai.note_text(note), facts)
        pills = [
            ("ok", "Numbers match the facts") if not rv["ungrounded"] else ("bad", "Unverified figures: " + ", ".join(rv["ungrounded"])),
            ("ok", "Consistent with decision") if not rv["contradiction"] else ("bad", rv["contradiction"]),
            ("ok", "No protected attributes") if not rv["protected"] else ("bad", "Mentions: " + ", ".join(rv["protected"])),
        ]
        if getattr(note, "injection_detected", False):
            pills.append(("bad", "AI flagged instructions in the input"))
        st.markdown("Automatic checks on this text: " + " ".join(f"<span class='pill {c}'>{t}</span>" for c, t in pills), unsafe_allow_html=True)
        st.caption(f"Written by: {used}. The decision above was made by policy rules and the risk model, not by the AI. "
                   "AI text can be wrong; the checks above flag the most common problems.")

        # ---------------- Q&A
        st.markdown("**Ask about this application**")
        with st.form("ask", clear_on_submit=True):
            q = st.text_input("Question", placeholder="e.g. What is the most this applicant could borrow?", label_visibility="collapsed")
            asked = st.form_submit_button("Ask")
        if asked and q.strip():
            if not ai.in_scope(q) or ai.injection_phrases(q):
                ss.qa.append((q, "I can only answer questions about this loan application and its result. "
                                 "The decision cannot be changed from here.", "scope filter (rules)"))
            elif outage:
                ss.qa.append((q, ai.fallback_answer(facts, q).answer, "rule-based template"))
            else:
                try:
                    ans, used_q = ai.answer_question(facts, q)
                    text = ans.answer
                    chk = ai.review(text, facts)
                    if chk["contradiction"] or chk["protected"]:
                        text, used_q = ai.fallback_answer(facts, q).answer, "rule-based template (AI answer failed checks)"
                    ss.qa.append((q, text, used_q))
                except ai.AIUnavailable:
                    ss.qa.append((q, ai.fallback_answer(facts, q).answer, "rule-based template"))
        for q_, a_, by in ss.qa[::-1]:
            st.markdown(f"> **Q:** {q_}\n>\n> {a_}\n>\n> <span class='small'>Answered by: {by}</span>", unsafe_allow_html=True)

        # ---------------- what-if
        st.divider()
        st.subheader("④ What-if")
        st.caption("Change the amount or tenure to see how the result moves. Uses rules and the model only (no AI call).")
        w1, w2 = st.columns(2)
        wl = w1.slider("Loan amount (Rs)", MIN_LOAN, MAX_LOAN, int(a["loan_amount"]), 10000, key="wl_" + cur["hash"])
        wt = w2.select_slider("Tenure (months)", TENURES, a["tenure_months"], key="wt_" + cur["hash"])
        s3, _ = score_one({**a, "loan_amount": wl, "tenure_months": wt}, rate_)
        st.markdown(
            f"<div class='decision {s3.decision}'>{DECISION_ICON[s3.decision]} <b>{s3.decision}</b> at {inr(wl)} for {wt} months "
            f"&nbsp;·&nbsp; EMI {inr(s3.new_emi)} &nbsp;·&nbsp; FOIR {s3.foir_after:.1%} &nbsp;·&nbsp; grade {s3.grade} ({s3.pd:.1%})</div>",
            unsafe_allow_html=True)

        # ---------------- export
        rep = io.StringIO()
        rep.write(f"CreditLens pre-screening report\nGenerated {datetime.now():%d %b %Y %H:%M}\n\n")
        rep.write(f"Applicant: {cur['name'] or '(no name)'}\nDecision: {s.decision} (decided by: {s.decided_by})\n")
        rep.write(f"Risk grade {s.grade}, default probability {s.pd:.1%}\nLoan {inr(a['loan_amount'])}, {a['tenure_months']} months at {rate_}%\n")
        rep.write(f"New EMI {inr(s.new_emi)}; FOIR {s.foir_before:.0%} -> {s.foir_after:.0%}; max eligible {inr(s.max_eligible)}\n\nPolicy rules:\n")
        for h in s.policy_hits or [("-", "none triggered", "")]:
            rep.write(f"  {h[0]} {h[1]} {h[2]}\n")
        rep.write("\nTop drivers:\n")
        for r in drivers.head(5).itertuples():
            rep.write(f"  {r.factor}: {r.shown_value} (avg {r.shown_avg}) - {r.direction}\n")
        rep.write(f"\nExplanation ({used}):\n{note.officer_summary}\n\nApplicant message:\n{note.applicant_message}\n")
        rep.write("\nPre-screening only. Final sanction needs document verification and a bureau pull.\n")
        st.download_button("⬇️ Download screening report (.txt)", rep.getvalue(), file_name=f"creditlens_{cur['hash']}.txt")

# ---------------------------------------------------------------- tab 2
with tab2:
    st.subheader("Batch screening")
    st.markdown("Upload a CSV of applicants to pre-screen a whole list at once. Rules and the risk model run on every row; "
                "no AI call is made per row, which keeps it fast and free. Open any applicant in the single view for an explanation.")
    tmpl = pd.read_csv("data/sample_batch.csv")
    cA, cB = st.columns([1, 1])
    cA.download_button("⬇️ Sample batch file (25 applicants)", tmpl.to_csv(index=False), "sample_batch.csv")
    up = cB.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")
    use_sample = st.button("Screen the sample batch")
    bdf = None
    if up is not None:
        try:
            bdf = pd.read_csv(up)
        except Exception as e:
            st.error(f"Could not read the file as CSV ({e}).")
    elif use_sample or ss.get("batch_on"):
        bdf = tmpl
        ss.batch_on = True
    if bdf is not None:
        need = ["age", "employment", "years_in_job", "monthly_income", "existing_emi", "loan_amount", "tenure_months",
                "credit_score", "enquiries_6m", "dpd30_24m", "card_utilisation", "home_owner"]
        miss = [c for c in need if c not in bdf.columns]
        if miss:
            st.error("Missing columns: " + ", ".join(miss))
        else:
            rows = []
            for i, r in bdf.iterrows():
                a = {c: r[c] for c in need}
                a["dpd90_24m"] = int(r.get("dpd90_24m", 0) or 0)
                a["credit_score"] = None if pd.isna(a["credit_score"]) else float(a["credit_score"])
                a["tenure_months"] = int(a["tenure_months"]) if not pd.isna(a["tenure_months"]) else None
                errs = validate(a)
                ref = r.get("applicant_ref", f"row {i + 1}")
                if errs:
                    rows.append(dict(ref=ref, decision="Invalid input", grade="", pd="", foir="", reason=errs[0]))
                    continue
                s, drv = score_one(a, rate)
                why = "; ".join(h[0] + " " + h[1] for h in s.policy_hits[:2]) or f"Model grade {s.grade}; top driver: {drv.iloc[0].factor}"
                rows.append(dict(ref=ref, decision=s.decision, grade=s.grade, pd=f"{s.pd:.1%}", foir=f"{s.foir_after:.0%}", reason=why))
            res = pd.DataFrame(rows)
            cnt = res["decision"].value_counts()
            mm = st.columns(4)
            for col, lab in zip(mm, ["Approve", "Refer", "Reject", "Invalid input"]):
                col.metric(lab, int(cnt.get(lab, 0)))
            st.dataframe(res.rename(columns={"ref": "Applicant", "decision": "Decision", "grade": "Grade", "pd": "Default prob.",
                                             "foir": "FOIR after", "reason": "Main reason"}), hide_index=True, width="stretch", height=420)
            st.download_button("⬇️ Download results (.csv)", res.to_csv(index=False), "creditlens_batch_results.csv")

# ---------------------------------------------------------------- tab 3
with tab3:
    st.subheader("Session history")
    if ss.history:
        st.dataframe(pd.DataFrame(ss.history).drop(columns=["hash"]).rename(columns={
            "time": "Time", "name": "Applicant", "loan": "Loan", "decision": "Decision", "grade": "Grade", "pd": "Default prob."}),
            hide_index=True, width="stretch")
        if st.button("Clear history"):
            ss.history, ss.notes_cache = [], {}
            ss.pop("current", None)
            st.rerun()
    else:
        st.write("No applications screened yet in this session.")
    st.caption("History lives only in this browser session. Re-submitting the same application does not create a duplicate "
               "or a new AI call. Refreshing the page starts a new session.")

# ---------------------------------------------------------------- tab 4
with tab4:
    st.subheader("How CreditLens decides")
    st.markdown(
        "1. **Input checks** (rules): missing, out-of-range or impossible values are rejected before anything else runs.\n"
        "2. **Lending-policy rules** (rules): hard limits a lender would apply, such as minimum income, credit score and EMI burden.\n"
        "3. **Risk model** (machine learning): a logistic regression estimates the probability of default and shows "
        "how much each factor pushed it up or down.\n"
        "4. **Decision** (rules): the stricter of the policy outcome and the model view wins.\n"
        "5. **Explanation** (Gemini): plain-language notes for the officer and applicant, checked by code for invented numbers, "
        "contradictions and protected attributes. If Gemini is down, a rule-based note is used.")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Policy rules**")
        st.dataframe(pd.DataFrame(POLICY_TEXT, columns=["Code", "Rule", "Outcome"]), hide_index=True, width="stretch")
    with c2:
        st.markdown("**Risk grades**")
        lo = 0.0
        g = []
        for ub, gr, view in GRADES:
            g.append((gr, f"{lo:.0%} to {min(ub, 1):.0%}", view))
            lo = ub
        st.dataframe(pd.DataFrame(g, columns=["Grade", "Default probability", "Model view"]), hide_index=True, width="stretch")
        st.markdown(f"**Model card.** Logistic regression on {MODEL.n_train:,} synthetic applicants (default rate "
                    f"{MODEL.base_rate:.1%}), holdout AUC {MODEL.auc:.2f}. The data is generated, not real, so the scores "
                    "show how the method works; they are not calibrated to any real lender's book.")
    st.markdown("**Not used, by design:** gender, religion, caste, marital status and home city. They are not collected, "
                "so neither the model nor the AI can use them.")
