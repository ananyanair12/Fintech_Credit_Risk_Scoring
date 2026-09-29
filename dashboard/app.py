"""Analyst dashboard (Streamlit).

Run locally:  streamlit run dashboard/app.py
Needs the API running for the 'Applicant explorer' tab (API_URL, default http://localhost:8000).
"""
import json
import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
IMAGES = ROOT / "docs" / "images"
API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Credit Risk Analyst Dashboard", layout="wide")
st.title("Credit Risk Scoring - Analyst Dashboard")


def show_image(name: str, caption: str) -> None:
    path = IMAGES / name
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.info(f"{name} not found - run the pipeline to generate it.")


tab_models, tab_shap, tab_explore = st.tabs(
    ["Model comparison", "Global explainability", "Applicant explorer"]
)

# ------------------------------------------------------------------ model comparison
with tab_models:
    comp_path = ARTIFACTS / "model_comparison.csv"
    meta_path = ARTIFACTS / "metadata.json"
    if not comp_path.exists():
        st.warning("No results yet. Run `python -m src.pipeline` first.")
    else:
        comp = pd.read_csv(comp_path)
        st.subheader("Validation-set comparison (all MLflow runs)")
        show_cols = ["run_name", "val_auc", "val_gini", "val_ks", "train_auc", "train_time_s"]
        st.dataframe(comp[show_cols].round(4), use_container_width=True, hide_index=True)

        chart = px.bar(
            comp.sort_values("val_gini"), x="val_gini", y="run_name",
            orientation="h", title="Validation Gini by model", text_auto=".3f",
        )
        st.plotly_chart(chart, use_container_width=True)

        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            st.subheader(f"Selected model: {meta['model_name']}")
            t = meta["test_metrics"]
            at = meta["test_at_threshold"]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Test Gini", f"{t['gini']:.3f}")
            c2.metric("Test KS", f"{t['ks']:.3f}")
            c3.metric("Test AUC", f"{t['auc']:.3f}")
            c4.metric("p95 model latency", f"{meta['inference_latency_ms']['p95_ms']:.1f} ms")

            st.subheader("Decision threshold")
            st.write(
                f"Cost ratio assumed: **{meta['cost_ratio']:g} : 1** (missed default : wrongly declined "
                f"applicant). Chosen threshold on validation: **{meta['threshold']:.4f}**. "
                f"On the test set this gives precision {at['precision']:.2f}, recall {at['recall']:.2f}, "
                f"approval rate {at['approval_rate']:.1%}."
            )
            col_a, col_b = st.columns(2)
            with col_a:
                show_image("pr_curve.png", "Precision-recall curve")
            with col_b:
                show_image("threshold_cost.png", "Cost vs threshold")

# ------------------------------------------------------------------------ global SHAP
with tab_shap:
    shap_path = ARTIFACTS / "shap_global.csv"
    if not shap_path.exists():
        st.warning("No SHAP results yet. Run `python -m src.explain.shap_utils`.")
    else:
        g = pd.read_csv(shap_path).head(15).sort_values("mean_abs_shap")
        st.plotly_chart(
            px.bar(g, x="mean_abs_shap", y="feature", orientation="h",
                   title="Global feature importance (mean |SHAP|)"),
            use_container_width=True,
        )
        show_image("shap_summary.png", "SHAP summary (validation sample)")
        c1, c2, c3 = st.columns(3)
        with c1:
            show_image("shap_approved.png", "Example: clearly approved")
        with c2:
            show_image("shap_borderline.png", "Example: borderline")
        with c3:
            show_image("shap_declined.png", "Example: clearly declined")

# ------------------------------------------------------------------ applicant explorer
PRESETS = {
    "Typical applicant": dict(util=0.35, age=42, p3059=0, debt=0.42, income=5400, lines=8, p90=0, re=1, p6089=0, dep=2),
    "Low risk": dict(util=0.05, age=55, p3059=0, debt=0.15, income=9000, lines=6, p90=0, re=2, p6089=0, dep=1),
    "High risk": dict(util=1.10, age=29, p3059=3, debt=0.90, income=2200, lines=12, p90=2, re=0, p6089=2, dep=3),
}

with tab_explore:
    st.subheader("Score an applicant")
    preset_name = st.selectbox("Start from a preset", list(PRESETS))
    p = PRESETS[preset_name]
    k = preset_name  # keys change with the preset so defaults refresh

    c1, c2, c3 = st.columns(3)
    with c1:
        util = st.number_input("Revolving utilization", 0.0, 50.0, float(p["util"]), 0.05, key=f"u{k}")
        age = st.number_input("Age", 18, 120, int(p["age"]), key=f"a{k}")
        debt = st.number_input("Debt ratio (or debt amount if income unknown)", 0.0, 1e6, float(p["debt"]), 0.05, key=f"d{k}")
    with c2:
        income_known = st.checkbox("Monthly income known", True, key=f"ik{k}")
        income = st.number_input("Monthly income", 0.0, 1e7, float(p["income"]), 100.0, key=f"i{k}")
        lines = st.number_input("Open credit lines", 0, 100, int(p["lines"]), key=f"l{k}")
        re_loans = st.number_input("Real-estate loans", 0, 100, int(p["re"]), key=f"r{k}")
    with c3:
        p3059 = st.number_input("Times 30-59 days past due", 0, 98, int(p["p3059"]), key=f"p1{k}")
        p6089 = st.number_input("Times 60-89 days past due", 0, 98, int(p["p6089"]), key=f"p2{k}")
        p90 = st.number_input("Times 90+ days late", 0, 98, int(p["p90"]), key=f"p3{k}")
        dep = st.number_input("Dependents", 0, 30, int(p["dep"]), key=f"dp{k}")

    if st.button("Score applicant", type="primary"):
        payload = {
            "revolving_utilization": util, "age": int(age), "past_due_30_59": int(p3059),
            "debt_ratio": debt, "monthly_income": income if income_known else None,
            "open_credit_lines": int(lines), "past_due_90": int(p90),
            "real_estate_loans": int(re_loans), "past_due_60_89": int(p6089),
            "dependents": int(dep),
        }
        try:
            r = requests.post(f"{API_URL}/predict", json=payload, timeout=15)
            r.raise_for_status()
            out = r.json()
        except requests.RequestException as e:
            st.error(f"Could not reach the API at {API_URL}: {e}")
        else:
            m1, m2, m3 = st.columns(3)
            m1.metric("Risk score", f"{out['risk_score']:.3f}")
            m2.metric("Decision", out["decision"].upper())
            m3.metric("Threshold", f"{out['threshold']:.3f}")
            if out["decision"] == "decline":
                st.error("Score is at or above the decline threshold.")
            else:
                st.success("Score is below the decline threshold.")

            f = pd.DataFrame(out["shap_explanation"]["top_factors"])
            f["label"] = f["feature"] + " = " + f["value"].round(3).astype(str)
            fig = px.bar(
                f.sort_values("shap"), x="shap", y="label", orientation="h",
                color="direction",
                color_discrete_map={"increases_risk": "#d62728", "decreases_risk": "#2ca02c"},
                title="Why this score? (SHAP, log-odds; right = higher risk)",
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"API latency: {out['latency_ms']} ms | model: {out['model_version']}")