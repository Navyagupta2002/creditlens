# CreditLens: AI-assisted loan pre-screener

End-term project, **AI for Managers: Applications & Strategy** (PGDM 2025-27, FORE School of Management).
Use Case 5: Loan eligibility / credit-risk pre-screener · Format: App
Navya Gupta · Roll No. 065015

**Live app:** https://creditlens-navya.streamlit.app

## What it does
A loan officer enters an applicant (or picks one of three samples) and gets, in under a minute:
- an **Approve / Refer / Reject** pre-screening result,
- a risk grade (A to E) and an estimated probability of default from a logistic-regression model,
- the drivers behind the score, the lending-policy rules that fired, and a banker's rule-of-thumb cross-check,
- a plain-language explanation for the officer and the applicant, written by **Google Gemini** and checked by code,
- a what-if slider, batch screening of a CSV, session history and a downloadable report.

**Rules decide, the model scores, Gemini explains.** Gemini never sees the applicant's name and cannot change the decision.

## Files
| File | Purpose |
|---|---|
| `app.py` | Streamlit interface |
| `rules.py` | input validation, lending-policy rules (P1-P10), decision logic, facts sent to the AI |
| `model.py` | logistic-regression risk model and per-factor drivers |
| `ai.py` | Gemini calls (JSON schema), guardrail checks, rule-based fallback |
| `make_data.py` | generates the synthetic training data (fixed seed) |
| `data/` | 25-row sample batch; the synthetic training file is generated on first run from a fixed seed |
| `tests/run_tests.py` | 25 scenario tests; writes `tests/test_log.csv` |

All data is synthetic. No real applicant data is used.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
Add `GEMINI_API_KEY` to `.streamlit/secrets.toml` (see `secrets.toml.example`). Without a key the app still works and uses its rule-based note.

Pre-screening only. Not a credit decision or financial advice.
