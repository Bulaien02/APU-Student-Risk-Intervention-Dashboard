# path: notebooks/app.py
from __future__ import annotations

import os, sys, json, base64, random, html, io, datetime, time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from collections.abc import Mapping
from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────────────────────────────────────
# Page config
st.set_page_config(page_title="APU Predictive Academic Intervention Dashboard", page_icon="🎓", layout="wide")

# ──────────────────────────────────────────────────────────────────────────────
# Project root detection
def _find_project_root(start: Path) -> Path:
    markers = {".streamlit", "artifacts_models", "artifacts_split", "docs", "assets"}
    cur = start
    for _ in range(6):
        if any((cur / m).exists() for m in markers):
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    return start

APP_PATH = Path(__file__).resolve()
APP_DIR  = APP_PATH.parent
ROOT_DIR = _find_project_root(APP_DIR)

def _first_existing(*paths: Path):
    for p in paths:
        if p.exists():
            return p
    return None

ASSETS_DIR = _first_existing(APP_DIR / "assets", ROOT_DIR / "assets") or (APP_DIR / "assets")
DOCS_DIR = (ROOT_DIR / "docs") if (ROOT_DIR / "docs").exists() else (APP_DIR / "docs")


# ──────────────────────────────────────────────────────────────────────────────
# Core imports
from intervention_core import (  # noqa: E402
    load_artifacts,
    ensure_docindex_template,
    audit_docindex, load_docindex,
    generate_intervention_for_row,
    generate_intervention_for_index,
)

# ──────────────────────────────────────────────────────────────────────────────
# Secrets helpers
def _find_key_recursive(mapping: Mapping, targets: tuple[str, ...], path: str = ""):
    for k, v in mapping.items():
        new_path = f"{path}.{k}" if path else str(k)
        if str(k) in targets and v:
            return new_path, str(k), v
        if isinstance(v, Mapping):
            found = _find_key_recursive(v, targets, new_path)
            if found:
                return found
    return None

def load_gemini_secret() -> Tuple[bool, str, str, bool]:
    key_source = ""
    secrets_file = ROOT_DIR / ".streamlit" / "secrets.toml"
    secrets_file_found = secrets_file.exists()

    try:
        if hasattr(st, "secrets"):
            found = _find_key_recursive(st.secrets, ("GEMINI_API_KEY", "GOOGLE_API_KEY"))
            if found:
                dotted, _, value = found
                os.environ["GEMINI_API_KEY"] = str(value)
                key_source = f"st.secrets:{dotted}"
            model_found = _find_key_recursive(st.secrets, ("GEMINI_MODEL",))
            if model_found:
                _, _, model_value = model_found
                os.environ["GEMINI_MODEL"] = str(model_value)
    except Exception:
        pass

    if not os.environ.get("GEMINI_API_KEY") and secrets_file_found:
        try:
            try:
                import tomllib as _toml
            except Exception:
                import tomli as _toml
            data = _toml.loads(secrets_file.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                found = _find_key_recursive(data, ("GEMINI_API_KEY", "GOOGLE_API_KEY"))
                if found:
                    dotted, _, value = found
                    os.environ["GEMINI_API_KEY"] = str(value)
                    key_source = f"secrets.toml:{dotted}"
                model_found = _find_key_recursive(data, ("GEMINI_MODEL",))
                if model_found:
                    _, _, model_value = model_found
                    os.environ["GEMINI_MODEL"] = str(model_value)
        except Exception:
            pass

    if not key_source and os.environ.get("GEMINI_API_KEY"):
        key_source = "env:GEMINI_API_KEY"
    elif not key_source and os.environ.get("GOOGLE_API_KEY"):
        os.environ["GEMINI_API_KEY"] = os.environ["GOOGLE_API_KEY"]
        key_source = "env:GOOGLE_API_KEY"

    has = bool(os.environ.get("GEMINI_API_KEY"))
    model_name = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash" if has else "fallback-template")

    try:
        import google.generativeai as _  # noqa: F401
        genai_installed = True
    except Exception:
        genai_installed = False

    st.session_state["_secrets_file_found"] = secrets_file_found
    return has, model_name, key_source or "(none)", genai_installed

HAS_GEMINI, GEM_MODEL, GEM_SRC, GENAI_OK = load_gemini_secret()

# Prefer Gemini globally when available
FORCE_GEMINI = bool(HAS_GEMINI and GENAI_OK)
if FORCE_GEMINI:
    os.environ.setdefault("PREFER_GEMINI", "1")

# ──────────────────────────────────────────────────────────────────────────────
# Boot artifacts + doc index + thresholds
@st.cache_data(show_spinner=False)
def boot():
    X_test, model, label_map, feature_order = load_artifacts()

    cfg = DOCS_DIR / "docindex.json"
    if not cfg.exists():
        ensure_docindex_template(cfg)

    docx = None
    if cfg.exists():
        audit_docindex(cfg, strict=False)
        docx = load_docindex(cfg)

    norm_meta = {}
    p = ROOT_DIR / "artifacts_models" / "normalization_meta.json"
    if p.exists():
        try:
            norm_meta = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            norm_meta = {}
    return X_test, model, label_map, feature_order, docx, norm_meta

X_test, model, label_map, feature_order, docx, norm_meta = boot()

def load_thresholds() -> Dict[str, float]:
    default = {"high": 0.60, "medium": 0.35}
    cfg = ROOT_DIR / ".streamlit" / "thresholds.toml"
    if not cfg.exists():
        return default
    try:
        try:
            import tomllib as _toml
        except Exception:
            import tomli as _toml
        data = _toml.loads(cfg.read_text(encoding="utf-8"))
        out = {**default}
        for k in ("high", "medium"):
            if k in data:
                out[k] = float(data[k])
        return out
    except Exception:
        return default

THRESH = load_thresholds()

# ──────────────────────────────────────────────────────────────────────────────
# Gemini-preferred wrappers
def _safe_call_generate_row(X_one, X_test, model, label_map, doc_index, tone):
    try:
        return generate_intervention_for_row(
            X_one, X_test, model, label_map,
            doc_index=doc_index, tone=tone, prefer_gemini=True
        )
    except TypeError:
        return generate_intervention_for_row(
            X_one, X_test, model, label_map, doc_index=doc_index, tone=tone
        )

def _safe_call_generate_index(i, X, model, label_map, doc_index, tone, k_policy=5, k_features=3):
    try:
        return generate_intervention_for_index(
            i=i, X=X, model=model, label_map=label_map,
            doc_index=doc_index, tone=tone,
            k_policy=k_policy, k_features=k_features, prefer_gemini=True
        )
    except TypeError:
        return generate_intervention_for_index(
            i=i, X=X, model=model, label_map=label_map,
            doc_index=doc_index, tone=tone,
            k_policy=k_policy, k_features=k_features
        )

def _prefer_gemini_row(X_one, X_test, model, label_map, doc_index, tone, tries: int = 3):
    last = None
    for _ in range(max(1, tries)):
        if FORCE_GEMINI:
            os.environ["PREFER_GEMINI"] = "1"
        out = _safe_call_generate_row(X_one, X_test, model, label_map, doc_index, tone)
        last = out
        if not FORCE_GEMINI:
            return out
        if str(out.get("llm_source", "")).lower() == "gemini":
            return out
    return last

def _prefer_gemini_index(i, X, model, label_map, doc_index, tone, k_policy=5, k_features=3, tries: int = 3):
    last = None
    for _ in range(max(1, tries)):
        if FORCE_GEMINI:
            os.environ["PREFER_GEMINI"] = "1"
        out = _safe_call_generate_index(i, X, model, label_map, doc_index, tone, k_policy, k_features)
        last = out
        if not FORCE_GEMINI:
            return out
        if str(out.get("llm_source", "")).lower() == "gemini":
            return out
    return last

# ──────────────────────────────────────────────────────────────────────────────
# Header (logos + compact style)
def _find_logo(keyword: str) -> Optional[Path]:
    candidates = []
    for base in [ASSETS_DIR, ASSETS_DIR / "branding", ROOT_DIR]:
        if base.exists():
            for p in base.rglob("*"):
                if p.is_file():
                    name = p.name.lower()
                    if ("logo" in name) and (keyword in name):
                        candidates.append(p)
    ext_rank = {".png": 0, ".jpg": 1, ".jpeg": 2, ".svg": 3}
    candidates.sort(key=lambda p: ext_rank.get(p.suffix.lower(), 9))
    return candidates[0] if candidates else None

def _b64(img_path: Path) -> str:
    try:
        return base64.b64encode(img_path.read_bytes()).decode("utf-8")
    except Exception:
        return ""

# Header (logos + compact style)
def header():
    apu = _find_logo("apu")
    dmu = _find_logo("dmu")
    apu_b64 = _b64(apu) if apu else ""
    dmu_b64 = _b64(dmu) if dmu else ""

    llm_text = f"LLM: Gemini ({GEM_MODEL})" if HAS_GEMINI else "LLM: fallback template (no key)"

    # Styles (centered title with robust 3-column grid)
    st.markdown(
        """
        <style>
          :root { --card-bg: rgba(255,255,255,.04); --line: rgba(255,255,255,.08); }

          /* 3-column grid: left | center(auto) | right; center stays truly centered */
          .brand{
            display:grid;
            grid-template-columns: 1fr auto 1fr;
            align-items:center;
            column-gap:12px;
            padding:12px 16px;
            border-radius:14px;
            background:var(--card-bg);
            border:1px solid var(--line);
            margin-bottom:10px;
          }
          .brand > :first-child{ justify-self:start; }
          .brand > :nth-child(2){ justify-self:center; text-align:center; }
          .brand > :last-child{
            justify-self:end;
            display:flex; align-items:center; gap:10px; max-width:100%;
          }

          .badge{
            background:rgba(255,255,255,.98);
            border-radius:12px; padding:6px 10px;
            box-shadow:0 6px 22px rgba(0,0,0,.18), inset 0 1px 0 rgba(255,255,255,.85);
          }
          .logo {height:50px; width:auto; object-fit:contain}

          .chip {
            border:1px solid var(--line);
            padding:4px 10px; border-radius:999px;
            font-size:.75rem; opacity:.95; white-space:nowrap;
          }
          .ok   {background:rgba(76,175,80,.18)}
          .warn {background:rgba(255,152,0,.18)}

          /* ✨ Decorative bits */
          .title-gradient{
            margin:0 0 2px 0; font-weight:800; font-size:1.6rem;
            background:linear-gradient(90deg,#e8f0ff,#b1c6ff,#9cf1e0);
            -webkit-background-clip:text; background-clip:text; color:transparent;
            text-shadow:0 0 18px rgba(128,170,255,.18);
          }
          .chip.pulse{ animation:pulse 2.4s ease-in-out infinite; }
          @keyframes pulse{
            0%,100%{ box-shadow:0 0 0 rgba(99,179,237,0); }
            50%{ box-shadow:0 0 18px rgba(99,179,237,.35); }
          }

          @media (max-width:780px){
            .brand{ grid-template-columns:1fr; row-gap:8px; text-align:center; }
            .brand > :first-child, .brand > :last-child{ justify-self:center; }
            .chip{ white-space:normal; }
          }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="brand">
          <div class="badge">
            {('<img class="logo" src="data:image/*;base64,'+ apu_b64 + '">') if apu_b64 else ''}
          </div>

          <div>
            <h1 class="title-gradient">
              APU Predictive Academic Intervention Dashboard
            </h1>
            <p style="margin:0; font-size:.85rem; opacity:.8">
              RAG • SHAP/LIME • Gemini • XGBoost
            </p>
          </div>

          <div>
            <span class="chip pulse {'ok' if (HAS_GEMINI and GENAI_OK) else 'warn'}">{llm_text}</span>
            <div class="badge">
              {('<img class="logo" src="data:image/*;base64,'+ dmu_b64 + '">') if dmu_b64 else ''}
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

header()

# ──────────────────────────────────────────────────────────────────────────────
# Extra styling
st.markdown("""
<style>
.banner{
  display:flex; align-items:center; justify-content:space-between;
  padding:14px 16px; border-radius:14px;
  background:linear-gradient(90deg, rgba(64,129,255,.18), rgba(8,180,155,.18));
  border:1px solid rgba(255,255,255,.10); margin:8px 0 14px;
}
.banner .left{font-size:1.8rem; line-height:1}
.banner .title{font-weight:800; font-size:1.6rem; margin-bottom:2px}
.banner .sub{opacity:.75; font-size:.85rem}
.banner .pct{font-weight:900; font-size:1.6rem; padding:.2rem .6rem; border-radius:10px;
  background:rgba(255,255,255,.06); border:1px solid rgba(255,255,255,.10);}
.pills{display:flex;flex-wrap:wrap;gap:8px;margin-top:6px}
.pill{display:inline-flex;align-items:center;gap:6px;padding:6px 10px;border-radius:999px;
  border:1px solid rgba(255,255,255,.12); background:rgba(255,255,255,.04);
  font-size:.85rem}
.pill:before{content:""; width:8px; height:8px; border-radius:999px; display:inline-block}
.pill.shap:before{background:#6aa1ff}
.pill.lime:before{background:#8bd4c0}
.pill:hover{box-shadow:0 6px 22px rgba(0,0,0,.15) inset}
.badge-src{padding:4px 8px;border:1px solid rgba(255,255,255,.25);border-radius:999px;margin-left:8px;opacity:.9}
.badge-high{background:rgba(244, 67, 54,.18)}
.badge-med{background:rgba(255, 152, 0,.18)}
.badge-low{background:rgba(76, 175, 80,.18)}

/* Primary button micro-interactions */
div.stButton > button[kind="primary"]{
  transition: transform .08s ease, box-shadow .08s ease;
  box-shadow: 0 8px 24px rgba(0,0,0,.18);
}
div.stButton > button[kind="primary"]:hover{
  transform: translateY(-1px);
  box-shadow: 0 10px 28px rgba(0,0,0,.22);
}
</style>
""", unsafe_allow_html=True)

# Small "About" expander
# --- Professional About panel (replace your current expander) ---
def render_about_panel():
    hi = int(THRESH.get("high", 0.60) * 100)
    med = int(THRESH.get("medium", 0.35) * 100)
    llm_label = f"Gemini ({GEM_MODEL})" if (HAS_GEMINI and GENAI_OK) else "Fallback (no key detected)"

    with st.expander("ℹ️ About this dashboard", expanded=False):
        st.markdown(
            f"""
            ### APU Predictive Academic Intervention Dashboard
            This application helps staff **identify at-risk students early**, **explain the drivers** behind each prediction, and **generate tailored intervention plans** grounded in APU resources.

            **What you'll get on every run**
            - **Risk**: At-Risk / Average / Excellent, with probabilities  
            - **Why**: Top drivers (SHAP & LIME) in plain language  
            - **Action**: A structured, tone-aware plan (LLM: **{llm_label}**) plus relevant APU links  
            - **Operations**: Filter & triage, split by advisor, and export CSV/Excel
            """,
            unsafe_allow_html=False,
        )

        tab_overview, tab_quickstart, tab_methods, tab_govern = st.tabs(
            ["Overview", "Quick start", "How it works", "Data & governance"]
        )

        with tab_overview:
            col1, col2 = st.columns([1.2, 1])
            with col1:
                st.markdown(
                    f"""
                    **Who is this for**
                    - Academic advisors and programme leaders
                    - Student services and success teams

                    **Risk bands**
                    - **High**: ≥ {hi}% P(At-Risk)  
                    - **Medium**: ≥ {med}% and &lt; {hi}%  
                    - **Low**: &lt; {med}%  

                    **Key ideas**
                    - *Predict → Explain → Act*: model prediction, explainability for trust, and a concrete plan.
                    - Plans are **grounded** in APU resources (docs/links); the LLM drafts, you decide.
                    """
                )
            with col2:
                st.markdown(
                    """
                    **Main tabs**
                    - **Manual entry** – Try a single student profile and get a plan.
                    - **Demo (on-demand)** – Sample an existing row to see end-to-end output.
                    - **Triage** – Rank, filter, capacity-limit, split by advisor, export.
                    - **Scenario Compare** – Tweak key levers (attendance, level, etc.) and compare risk/plan side-by-side.
                    """
                )

        with tab_quickstart:
            st.markdown(
                """
                **1) Manual entry**
                - Select values for each field → **Predict & generate plan**  
                - Review probabilities, top drivers, and the generated plan  
                - Download CSV/Excel/JSON if needed

                **2) Triage**
                - Filter by programme/level/attendance → set **Capacity**  
                - (Optional) **Show reasons** to append top SHAP drivers  
                - **Split by advisor** and export

                **3) Scenario Compare**
                - Choose a base row → adjust scenario controls  
                - Use quick actions (toggle/cycle) or **Apply playbook**  
                - **Run compare** to see how risk and plan change
                """
            )

        with tab_methods:
            st.markdown(
                """
                **Model**
                - Supervised classifier (XGBoost); features include demographics, programme one-hots, and academic indicators.
                - Outputs class probabilities and predicted label.

                **Explainability**
                - **SHAP** highlights features pushing *toward* the At-Risk class.
                - **LIME** offers a complementary local explanation.
                - We show the **top-3** contributors for quick sense-making.

                **Intervention plans**
                - Generated with an LLM (Gemini when available).  
                - Plans are constrained to APU resources and study-skills guidance; advisors remain the final decision-makers.

                **Thresholds**
                - Risk bands are configurable via `.streamlit/thresholds.toml` (current High ≥ {hi}%, Medium ≥ {med}%).
                """
            )

        with tab_govern:
            st.markdown(
                """
                **Data handling**
                - The app does **not** persist raw student data unless you explicitly export or append to logs.
                - Optional advising logs are written to `logs/advising_log.csv` on demand.

                **API keys**
                - The LLM key is loaded from `st.secrets` or `.streamlit/secrets.toml`. If absent, a safe fallback template is used.

                **Good practice**
                - Use the tool to inform professional judgement, not replace it.
                - Treat outputs as decision support and review plans before sharing.

                **Support & maintenance**
                - Thresholds, labels, and resources are configurable (see `docs/` and `.streamlit/`).
                - If something looks off, please report the input, expected behaviour, and a screenshot/log snippet.
                """
            )

# Call this once where you previously had the simple expander:
render_about_panel()


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
def _years_to_z(age_years: int) -> float:
    if "AGE" in norm_meta and all(k in norm_meta["AGE"] for k in ("mean","std")):
        mu = float(norm_meta["AGE"]["mean"]); sd = float(norm_meta["AGE"]["std"]) or 1.0
        return (int(age_years) - mu)/sd
    a_min, a_max = 16, 60
    p = (int(age_years) - a_min) / (a_max - a_min)
    p = float(np.clip(p, 1e-3, 1-1e-3))
    return (p - 0.5) * 3.0

def _z_to_years(z: float) -> int:
    try:
        z = float(z)
    except Exception:
        z = 0.0
    if "AGE" in norm_meta and all(k in norm_meta["AGE"] for k in ("mean", "std")):
        mu = float(norm_meta["AGE"]["mean"])
        sd = float(norm_meta["AGE"]["std"]) or 1.0
        years = mu + z * sd
    else:
        a_min, a_max = 16, 60
        p = np.clip(z / 3.0 + 0.5, 0.0, 1.0)
        years = a_min + p * (a_max - a_min)
    return int(np.clip(round(years), 16, 60))

def _bucket(p_atrisk: float) -> tuple[str, str]:
    if p_atrisk >= THRESH["high"]:
        return "High", "badge-high"
    if p_atrisk >= THRESH["medium"]:
        return "Medium", "badge-med"
    return "Low", "badge-low"

def _banner(pred_label: str, proba_dict: Dict[str, float]):
    top = max(proba_dict, key=proba_dict.get)
    pct = proba_dict[top] * 100.0
    emoji = {"At-Risk":"⚠️","Average":"📊","Excellent":"🏆"}.get(top, "🎯")
    st.markdown(f"""
    <div class="banner">
      <div class="left">{emoji}</div>
      <div class="center">
        <div class="title">{html.escape(top)}</div>
        <div class="sub">Predicted class</div>
      </div>
      <div class="right"><span class="pct">{pct:.1f}%</span></div>
    </div>
    """, unsafe_allow_html=True)

def _bar(prob_dict: Dict[str, float]):
    items  = sorted(prob_dict.items(), key=lambda kv: kv[1])
    labels = [k for k,_ in items]
    vals   = [float(v)*100 for _,v in items]
    try:
        import plotly.graph_objects as go
        template = "plotly_dark" if st.get_option("theme.base") == "dark" else "plotly_white"
        fig = go.Figure(go.Bar(
            x=vals, y=labels, orientation="h",
            text=[f"{v:.1f}%" for v in vals], textposition="outside",
            cliponaxis=False,
            marker=dict(colorscale="Blues", color=vals, line=dict(width=0)),
            hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
        ))
        fig.update_layout(
            template=template, height=260, bargap=0.28,
            margin=dict(l=6, r=70, t=6, b=10),
            xaxis=dict(title="Probability (%)", range=[0,100],
                       gridcolor="rgba(127,127,127,0.15)"),
            yaxis=dict(title="", showgrid=False),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        return
    except Exception:
        pass
    fig, ax = plt.subplots(figsize=(10.5, 2.2))
    bars = ax.barh(labels, vals)
    ax.set_xlabel("Probability (%)"); ax.set_xlim(0, 100); ax.grid(axis="x", linestyle=":", alpha=0.25)
    for spine in ax.spines.values(): spine.set_visible(False)
    for bar, pct in zip(bars, vals):
        y = bar.get_y() + bar.get_height()/2
        ax.text(min(pct+1.5, 98), y, f"{pct:.1f}%", va="center", ha="left")
    st.pyplot(fig, use_container_width=True)

def _feature_pills(title: str, items: list[str], kind: str):
    if not items:
        st.caption("—"); return
    pills = "".join(f"<span class='pill {kind}' title='{html.escape(str(x))}'>{html.escape(str(x))}</span>"
                    for x in items)
    st.markdown(f"**{title}:**<div class='pills'>{pills}</div>", unsafe_allow_html=True)

def _risk_gauge(p_atrisk: float):
    """Plotly gauge for P(At-Risk) in [0,1]."""
    try:
        import plotly.graph_objects as go
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=max(0.0, min(100.0, p_atrisk*100)),
            number={'suffix': '%', 'font': {'size': 32}},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'thickness': 0.25},
                'steps': [
                    {'range': [0, THRESH['medium']*100], 'color': 'rgba(76,175,80,.25)'},
                    {'range': [THRESH['medium']*100, THRESH['high']*100], 'color': 'rgba(255,152,0,.25)'},
                    {'range': [THRESH['high']*100, 100], 'color': 'rgba(244,67,54,.30)'},
                ],
            }
        ))
        fig.update_layout(height=220, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    except Exception:
        pass

def _render_plan(plan: Dict[str, Any], tone: str, llm_source: str, dl_key: str):
    # Risk badge
    risk_text = str(plan.get("risk_level", "-")).strip().title()
    risk_cls_map = {"High": "badge-high", "Medium": "badge-med", "Low": "badge-low"}
    risk_cls = risk_cls_map.get(risk_text, "")

    # Resources list
    resources_html = ""
    for r in plan.get("resources", []):
        title = r.get("title", "")
        url = r.get("url")
        if url:
            resources_html += f'<li><a href="{url}" target="_blank">{html.escape(title)}</a></li>'
        else:
            resources_html += f"<li>{html.escape(title)}</li>"

    llm_badge = "Gemini" if str(llm_source).lower() == "gemini" else "Fallback"

    st.markdown(
        f"""
        <div style="padding:14px;border:1px solid rgba(255,255,255,.18);border-radius:12px;">

          <!-- Top: Tone + Model -->
          <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:6px">
            <span style="opacity:.85">Tone</span>
            <span class="chip">{tone.title()}</span>
            <span class="badge-src">{llm_badge}</span>
          </div>

          <!-- Risk level row (left aligned, its own line) -->
          <div style="display:flex;align-items:center;gap:10px;margin:4px 0 12px 0">
            <span style="opacity:.75">Risk level</span>
            <span class="chip {risk_cls}" style="font-weight:700">{html.escape(risk_text)}</span>
          </div>

          <!-- Sections -->
          <div style="opacity:.75;margin-bottom:4px">Why</div>
          <div style="margin-bottom:12px">{html.escape(plan.get('why_model_thinks_so',''))}</div>

          <div style="opacity:.75;margin-bottom:4px">Study plan</div>
          <ul style="margin-top:0">{''.join(f'<li>{html.escape(str(x))}</li>' for x in plan.get('study_plan',[]))}</ul>

          <div style="opacity:.75;margin:10px 0 4px">Follow-up actions</div>
          <ul style="margin-top:0">{''.join(f'<li>{html.escape(str(x))}</li>' for x in plan.get('follow_up_actions',[]))}</ul>

          <div style="opacity:.75;margin:10px 0 4px">Resources</div>
          <ul style="margin-top:0">{resources_html}</ul>

          <div style="opacity:.55;margin-top:10px;font-size:12px">
            Citations: {', '.join(html.escape(str(c)) for c in plan.get('citations', []))}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.download_button(
        "Download plan (JSON)",
        data=json.dumps(plan, indent=2).encode("utf-8"),
        file_name="intervention_plan.json",
        mime="application/json",
        use_container_width=True,
        key=dl_key,
    )


# 🎊 Confetti effect (CSS-only, auto-cleans)
def _confetti(n: int = 80, duration: float = 2.6):
    colors = ['#f44336','#ff9800','#ffeb3b','#4caf50','#2196f3','#9c27b0']
    pieces = []
    for _ in range(n):
        left = random.randint(0,100)
        delay = random.uniform(0,0.6)
        dur = random.uniform(duration*0.7, duration*1.2)
        size = random.randint(6,10)
        color = random.choice(colors)
        rotate = random.randint(-180,180)
        pieces.append(
            f'<span style="left:{left}vw; width:{size}px; height:{int(size*1.6)}px; '
            f'background:{color}; animation-duration:{dur}s; animation-delay:{delay}s; '
            f'transform: rotate({rotate}deg)"></span>'
        )
    html_confetti = f'''
    <div class="confetti">{''.join(pieces)}</div>
    <style>
      .confetti {{ position: fixed; inset: 0; pointer-events: none; overflow: hidden; z-index: 9999; }}
      .confetti span {{
          position: absolute; top: -10vh; opacity:.95; border-radius:2px;
          animation-name: confettiFall; animation-timing-function: linear; animation-fill-mode: forwards;
      }}
      @keyframes confettiFall {{
        0%   {{ transform: translateY(-10vh) rotate(0deg); }}
        100% {{ transform: translateY(110vh) rotate(360deg); }}
      }}
    </style>
    '''
    ph = st.empty()
    ph.markdown(html_confetti, unsafe_allow_html=True)
    time.sleep(duration + 0.8)
    ph.empty()

# Export helpers
def _flatten_for_csv(run_id: str, tone: str, llm_source: str,
                     pred: Dict[str, Any], plan: Dict[str, Any],
                     inputs_snapshot: Dict[str, Any] | None = None) -> pd.DataFrame:
    row: Dict[str, Any] = {
        "run_id": run_id, "tone": tone, "llm_source": llm_source,
        "predicted_label": pred.get("pred_label"),
    }
    for k, v in pred.get("proba_dict", {}).items():
        row[f"proba_{k}"] = float(v)
    row["shap_top3"] = " | ".join(pred.get("shap_top", []) if "shap_top" in pred else [])
    row["lime_top3"] = " | ".join(pred.get("lime_top", []) if "lime_top" in pred else [])
    row["plan_risk_level"] = plan.get("risk_level")
    row["plan_why"]        = plan.get("why_model_thinks_so")
    row["plan_study_plan"] = " | ".join(plan.get("study_plan", []))
    row["plan_followups"]  = " | ".join(plan.get("follow_up_actions", []))
    row["plan_resources"]  = " | ".join([f"{r.get('title','')}" + (f" ({r.get('url')})" if r.get("url") else "") for r in plan.get("resources", [])])
    row["plan_citations"]  = " | ".join(plan.get("citations", []))
    if inputs_snapshot:
        for k, v in inputs_snapshot.items():
            row[f"input_{k}"] = v
    return pd.DataFrame([row])

def _excel_bytes(pred_df: pd.DataFrame, plan: Dict[str, Any]) -> Optional[bytes]:
    try:
        engine = None
        try:
            import xlsxwriter  # noqa
            engine = "xlsxwriter"
        except Exception:
            try:
                import openpyxl  # noqa
                engine = "openpyxl"
            except Exception:
                return None
        bio = io.BytesIO()
        with pd.ExcelWriter(bio, engine=engine) as xl:
            pred_df.to_excel(xl, index=False, sheet_name="prediction")
            pd.DataFrame({
                "risk_level": [plan.get("risk_level")],
                "why_model_thinks_so": [plan.get("why_model_thinks_so")],
                "tone": [plan.get("tone")],
            }).to_excel(xl, index=False, sheet_name="plan")
            pd.DataFrame({"order": list(range(1, len(plan.get("study_plan", []))+1)),
                          "text": plan.get("study_plan", [])}).to_excel(xl, index=False, sheet_name="plan_items")
            pd.DataFrame(plan.get("resources", [])).to_excel(xl, index=False, sheet_name="resources")
            pd.DataFrame({"citation": plan.get("citations", [])}).to_excel(xl, index=False, sheet_name="citations")
        return bio.getvalue()
    except Exception:
        return None

def _excel_bytes_table(df: pd.DataFrame, sheet_name: str = "triage") -> Optional[bytes]:
    try:
        engine = None
        try:
            import xlsxwriter  # noqa
            engine = "xlsxwriter"
        except Exception:
            try:
                import openpyxl  # noqa
                engine = "openpyxl"
            except Exception:
                return None
        bio = io.BytesIO()
        with pd.ExcelWriter(bio, engine=engine) as xl:
            df.to_excel(xl, index=False, sheet_name=sheet_name)
        return bio.getvalue()
    except Exception:
        return None

def _split_by_advisors(df: pd.DataFrame, advisors: list[str], sort_col: str = "P_AtRisk") -> pd.DataFrame:
    if not advisors:
        return df.copy()
    d = df.copy()
    advisors = [a.strip() for a in advisors if a and a.strip()]
    if not advisors:
        return df.copy()
    if sort_col in d.columns:
        d = d.sort_values(sort_col, ascending=False)
    k = len(advisors)
    d["advisor_assigned"] = [advisors[i % k] for i in range(len(d))]
    d["queue_pos"] = d.groupby("advisor_assigned").cumcount() + 1
    return d.reset_index(drop=True)

def _excel_bytes_df(df: pd.DataFrame, sheet_name: str = "advising_log") -> bytes:
    bio = BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:   # or engine="xlsxwriter"
        df.to_excel(writer, index=False, sheet_name=sheet_name)
    bio.seek(0)
    return bio.getvalue()

# ──────────────────────────────────────────────────────────────────────────────
# Diagnostics
with st.expander("🧪 Diagnostics", expanded=False):
    def diag_chip(label: str, value: str, ok: Optional[bool]=None) -> str:
        cls = "chip ok" if ok is True else ("chip warn" if ok is False else "chip")
        return f"<span class='{cls}'><b>{html.escape(label)}</b>: {html.escape(str(value))}</span>"
    last_src = st.session_state.get("last_llm_source", "-")
    chips = [
        diag_chip("gemini_key_found", str(bool(HAS_GEMINI)).lower(), HAS_GEMINI),
        diag_chip("key_source", GEM_SRC or "(none)"),
        diag_chip("genai_installed", str(bool(GENAI_OK)).lower(), GENAI_OK),
        diag_chip("model", GEM_MODEL),
        diag_chip("secrets_file_found", str(bool(st.session_state.get("_secrets_file_found", False))).lower()),
        diag_chip("last_plan_source", last_src if last_src != "-" else "-"),
    ]
    st.markdown("<div class='diag'><div class='chips-row'>" + "".join(chips) + "</div></div>", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# Tone + tabs
with st.container():
    tone = st.selectbox("Tone", ["supportive", "coachy", "concise"], index=0)
    if not (HAS_GEMINI and GENAI_OK):
        st.caption("Gemini key/library not detected; plans will use fallback until available.")

tab_manual, tab_demo, tab_triage, tab_compare = st.tabs(
    ["✍️ Manual entry", "🧪 Demo (on-demand)", "🚨 Triage", "🧪 Scenario Compare"]
)

# ──────────────────────────────────────────────────────────────────────────────
# === MANUAL TAB ===
with tab_manual:
    gender_opt   = ["Male", "Female"]
    status_opt   = ["Active", "Inactive"]
    school_opt   = ["School of Computing (SoC)", "School of Technology (SoT)"]
    binYN        = ["Yes", "No"]
    att_opt      = ["Above 80%", "Below 80%"]
    origin_opt   = ["Local", "International"]
    level_opt    = ["Level 1", "Level 2", "Level 3"]
    entry_opt    = ["Certificate", "Foundation", "Diploma", "Year 1", "UEC", "STPM", "Pre-U", "High School", "Degree"]

    prog_cols  = [c for c in feature_order if c.startswith("Programme_")]
    prog_names = [c.replace("Programme_", "").replace("_"," ") for c in prog_cols]
    prog_map   = {disp: col for disp, col in zip(prog_names, prog_cols)}

    c1,c2,c3 = st.columns(3)
    with c1:
        gender = st.selectbox("Gender", gender_opt, index=None, placeholder="— Select —")
        prog_on = st.selectbox("Progression On Time", binYN, index=None, placeholder="— Select —")
        lower   = st.selectbox("Lower Level", ["No","Yes"], index=None, placeholder="— Select —")
        level   = st.selectbox("Level / Year", level_opt, index=None, placeholder="— Select —")
        programme = st.selectbox("Programme", sorted(prog_names), index=None, placeholder="— Select —")
    with c2:
        status  = st.selectbox("Student Status", status_opt, index=None, placeholder="— Select —")
        grad_on = st.selectbox("Graduation On Time", binYN, index=None, placeholder="— Select —")
        fin_aid = st.selectbox("Financial Aid", ["Yes","No"], index=None, placeholder="— Select —")
        entry_q = st.selectbox("Entry Qualification", entry_opt, index=None, placeholder="— Select —")
    with c3:
        school = st.selectbox("School", school_opt, index=None, placeholder="— Select —")
        att    = st.selectbox("Attendances", att_opt, index=None, placeholder="— Select —")
        origin = st.selectbox("Student Origin", origin_opt, index=None, placeholder="— Select —")

    st.caption("We do not ask for CGPA to avoid data leakage. It will be imputed internally if required.")
    age_pick = st.selectbox("Age (years)", [str(i) for i in range(16,61)], index=None, placeholder="— Select —")

    ready = all(x is not None for x in [gender, prog_on, lower, level, programme,
                                        status, grad_on, fin_aid, entry_q, school, att, origin, age_pick])

    btn = st.button("Predict & generate plan", type="primary", use_container_width=True, disabled=not ready)
    if not ready:
        st.info("Fill in all fields to enable the button.", icon="ℹ️")

    if btn and ready:
        row = pd.Series(0, index=feature_order, dtype="float64")
        gender_map  = {"Male":1, "Female":0}
        status_map  = {"Active":0, "Inactive":1}
        school_map  = {"School of Computing (SoC)":0, "School of Technology (SoT)":1}
        yes_no_std  = {"Yes":0, "No":1}
        att_map     = {"Above 80%":0, "Below 80%":1}
        lower_map   = {"No":0, "Yes":1}
        fin_map     = {"Yes":0, "No":1}
        origin_map  = {"Local":0, "International":1}
        level_map   = {"Level 1":0, "Level 2":1, "Level 3":2}
        entry_map   = {"Certificate":0, "Foundation":1, "Diploma":2, "Year 1":3,
                       "UEC":4, "STPM":5, "Pre-U":6, "High School":7, "Degree":8}

        row["GENDER"]               = gender_map[gender]
        row["STUDENT_STATUS"]       = status_map[status]
        row["SCHOOL"]               = school_map[school]
        row["PROGRESSION_ON_TIME"]  = yes_no_std[prog_on]
        row["GRADUATION_ON_TIME"]   = yes_no_std[grad_on]
        row["ATTENDANCES"]          = att_map[att]
        row["LOWER_LEVEL"]          = lower_map[lower]
        row["FINANCIAL_AID"]        = fin_map[fin_aid]
        row["STUDENT_ORIGIN"]       = origin_map[origin]
        row["Level/Year"]           = level_map[level]
        row["ENTRY_QUALIFICATION"]  = entry_map[entry_q]
        for c in prog_cols: row[c] = 0
        row[prog_map[programme]] = 1

        row["AGE"] = _years_to_z(int(age_pick))
        if "CGPA" in row.index:
            try: row["CGPA"] = float(np.median(X_test["CGPA"]))
            except Exception: row["CGPA"] = 0.0

        X_one = pd.DataFrame([row])
        try:
            out = _prefer_gemini_row(X_one, X_test, model, label_map, docx, tone)
            st.session_state["last_llm_source"] = out.get("llm_source", "fallback")
        except Exception as e:
            st.error(f"Plan generation failed: {e}", icon="🚫")
            st.stop()

        # ✨ little celebration + toast
        st.toast("Prediction ready", icon="✨")
        _confetti(n=100, duration=2.4)

        left, right = st.columns([1.2, 1])
        with left:
            st.markdown("### Predicted class & probabilities")
            _banner(out["prediction"]["pred_label"], out["prediction"]["proba_dict"])
            _risk_gauge(out["prediction"]["proba_dict"].get("At-Risk", 0.0))
            _bar(out["prediction"]["proba_dict"])
            st.markdown("### Top features")
            cA, cB = st.columns(2)
            with cA: _feature_pills("SHAP (top-3)", out.get("shap_top", []), "shap")
            with cB: _feature_pills("LIME (top-3)", out.get("lime_top", []), "lime")
        with right:
            st.markdown("### Personalized intervention plan")
            _render_plan(out["plan"], tone, out.get("llm_source","fallback"), dl_key=f"dl-{out['run_id']}")
            inputs_snapshot = {
                "Gender": gender, "Student_Status": status, "School": school,
                "Progression_On_Time": prog_on, "Graduation_On_Time": grad_on,
                "Attendances": att, "Lower_Level": lower, "Financial_Aid": fin_aid,
                "Student_Origin": origin, "Level_Year": level, "Entry_Qualification": entry_q,
                "Programme": programme, "Age_years": age_pick,
            }
            flat = _flatten_for_csv(out["run_id"], tone, out.get("llm_source","fallback"),
                                    {**out["prediction"], "shap_top": out.get("shap_top", []), "lime_top": out.get("lime_top", [])},
                                    out["plan"], inputs_snapshot)
            st.download_button("Download result (CSV)", data=flat.to_csv(index=False).encode("utf-8"),
                               file_name=f"result_{out['run_id']}.csv", mime="text/csv",
                               use_container_width=True, key=f"dl-csv-{out['run_id']}")
            xbytes = _excel_bytes(flat, out["plan"])
            if xbytes is not None:
                st.download_button("Download result (Excel)", data=xbytes,
                                   file_name=f"result_{out['run_id']}.xlsx",
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                   use_container_width=True, key=f"dl-xlsx-{out['run_id']}")
            else:
                st.caption("Install **openpyxl** or **xlsxwriter** to enable Excel export.")

# === DEMO TAB ===
with tab_demo:
    if "demo_idx" not in st.session_state:
        st.session_state.demo_idx = random.randrange(len(X_test))

    st.caption("Click **Generate demo** to run a sample from X_test. Gemini-first, fallback only if needed.")
    col_a, col_b, col_c = st.columns([1, 4, 2])
    with col_a:
        if st.button("🎲 Random", use_container_width=True, key="btn_demo_random"):
            st.session_state.demo_idx = random.randrange(len(X_test))
    with col_b:
        idx = st.selectbox(
            "Select student index",
            options=list(range(len(X_test))),
            index=int(st.session_state.demo_idx),
            format_func=lambda x: f"Student {x}",
            key="demo_idx_selectbox"
        )
        if idx != st.session_state.demo_idx:
            st.session_state.demo_idx = int(idx)
        st.caption(f"Selected: **Student {st.session_state.demo_idx}**")
    with col_c:
        run_demo = st.button("Generate demo", type="primary", use_container_width=True, key="btn_run_demo")

    if run_demo:
        try:
            out = _prefer_gemini_index(
                i=int(st.session_state.demo_idx),
                X=X_test, model=model, label_map=label_map, doc_index=docx, tone=tone,
                k_policy=5, k_features=3
            )
            st.session_state["last_llm_source"] = out.get("llm_source", "fallback")
        except Exception as e:
            st.error(f"Plan generation failed: {e}", icon="🚫")
            st.stop()

        left, right = st.columns([1.2, 1])
        with left:
            st.markdown("### Predicted class & probabilities")
            _banner(out["prediction"]["pred_label"], out["prediction"]["proba_dict"])
            _risk_gauge(out["prediction"]["proba_dict"].get("At-Risk", 0.0))
            _bar(out["prediction"]["proba_dict"])
            st.markdown("### Top features")
            cA, cB = st.columns(2)
            with cA: _feature_pills("SHAP (top-3)", out.get("shap_top", []), "shap")
            with cB: _feature_pills("LIME (top-3)", out.get("lime_top", []), "lime")
        with right:
            st.markdown("### Personalized intervention plan")
            _render_plan(out["plan"], tone, out.get("llm_source","fallback"), dl_key=f"dl-demo-{out['run_id']}")
            raw_feats = X_test.iloc[int(st.session_state.demo_idx)].to_dict()
            flat = _flatten_for_csv(out["run_id"], tone, out.get("llm_source","fallback"),
                                    {**out["prediction"], "shap_top": out.get("shap_top", []), "lime_top": out.get("lime_top", [])},
                                    out["plan"], inputs_snapshot=raw_feats)
            st.download_button("Download result (CSV)", data=flat.to_csv(index=False).encode("utf-8"),
                               file_name=f"result_{out['run_id']}.csv", mime="text/csv",
                               use_container_width=True, key=f"dl-csv-demo-{out['run_id']}")
            xbytes = _excel_bytes(flat, out["plan"])
            if xbytes is not None:
                st.download_button("Download result (Excel)", data=xbytes,
                                   file_name=f"result_{out['run_id']}.xlsx",
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                   use_container_width=True, key=f"dl-xlsx-demo-{out['run_id']}")
            else:
                st.caption("Install **openpyxl** or **xlsxwriter** to enable Excel export.")

# ──────────────────────────────────────────────────────────────────────────────
# TRIAGE compute & tab
@st.cache_data(show_spinner=True)
def compute_triage(
    X: pd.DataFrame,
    _model,
    label_map: Dict[int, str],
    feature_order: list[str]
) -> pd.DataFrame:
    proba = _model.predict_proba(X)
    classes = [label_map[int(c)] for c in _model.classes_]
    preds = pd.DataFrame(proba, columns=classes, index=X.index)
    preds["predicted"] = preds[classes].idxmax(axis=1)
    preds["P_AtRisk"] = preds.get("At-Risk", pd.Series([0.0] * len(preds), index=preds.index))
    prog_cols = [c for c in feature_order if c.startswith("Programme_")]
    if prog_cols:
        names = [c.replace("Programme_", "").replace("_", " ") for c in prog_cols]
        argmax_idx = np.argmax(X[prog_cols].values, axis=1) if len(prog_cols) > 1 else np.zeros(len(X), dtype=int)
        vals = []
        for i, j in enumerate(argmax_idx):
            vals.append(names[int(j)] if X.iloc[i][prog_cols[int(j)]] == 1 else "Unknown")
        preds["Programme"] = vals
    else:
        preds["Programme"] = "Unknown"
    if "ATTENDANCES" in X.columns:
        preds["Attendance"] = ["Above 80%" if int(v) == 0 else "Below 80%" for v in X["ATTENDANCES"].values]
    else:
        preds["Attendance"] = "Unknown"
    if "Level/Year" in X.columns:
        rev = {0:"Level 1",1:"Level 2",2:"Level 3"}
        preds["Level"] = [rev.get(int(v), str(v)) for v in X["Level/Year"].values]
    else:
        preds["Level"] = "?"
    return preds

def _pretty_feat(name: str) -> str:
    if name.startswith("Programme_"):
        return "Programme: " + name.replace("Programme_", "").replace("_", " ")
    return name.replace("_", " ")

# ---------- SHAP helpers (robust + cached) ----------
@st.cache_resource(show_spinner=False)
def _build_shap_explainer(_model, background_df: pd.DataFrame):
    import shap
    try:
        return shap.TreeExplainer(_model), "TreeExplainer(model)"
    except Exception:
        pass
    try:
        booster = getattr(_model, "get_booster", lambda: None)()
        if booster is not None:
            return shap.TreeExplainer(booster), "TreeExplainer(booster)"
    except Exception:
        pass
    try:
        bg = background_df.sample(min(200, len(background_df)), random_state=0)
        return shap.Explainer(_model, bg), "Explainer(generic)"
    except Exception:
        raise

def _shap_values_for_target(explainer, X_batch: pd.DataFrame, model, label_map: dict, target_name: str = "At-Risk"):
    import numpy as np
    sv = None
    try:
        sv = explainer.shap_values(X_batch)
    except Exception:
        exp = explainer(X_batch)
        sv = getattr(exp, "values", exp)

    if isinstance(sv, list):
        label_to_idx = {v: k for k, v in label_map.items()}
        if target_name in label_to_idx and hasattr(model, "classes_"):
            target_code = label_to_idx[target_name]
            target_ix = list(model.classes_).index(target_code)
        else:
            target_ix = 0
        return np.asarray(sv[target_ix])

    vals = np.asarray(getattr(sv, "values", sv))
    if vals.ndim == 3:
        label_to_idx = {v: k for k, v in label_map.items()}
        if target_name in label_to_idx and hasattr(model, "classes_"):
            target_code = label_to_idx[target_name]
            target_ix = list(model.classes_).index(target_code)
        else:
            target_ix = 0
        return vals[:, target_ix, :]
    return vals
# ---------- /SHAP helpers ----------

with tab_triage:
    st.caption("Rank students by risk, filter, pick capacity, split by advisor, and export.")
    preds = compute_triage(X_test, model, label_map, feature_order)

    c1, c2, c3, c4 = st.columns([1.6, 1.2, 1.2, 1.2])
    with c1:
        f_prog = st.multiselect("Programme", sorted([p for p in preds["Programme"].unique() if p != "Unknown"]),
                                default=[], key="triage_prog")
    with c2:
        f_level = st.multiselect("Level", ["Level 1","Level 2","Level 3"], default=[], key="triage_level")
    with c3:
        f_att = st.multiselect("Attendance", ["Above 80%", "Below 80%"], default=[], key="triage_att")
    with c4:
        capacity = st.slider("Capacity (students)", 5, min(200, len(preds)), min(25, len(preds)), 1, key="triage_capacity")

    mask = pd.Series([True]*len(preds), index=preds.index)
    if f_prog:  mask &= preds["Programme"].isin(f_prog)
    if f_att:   mask &= preds["Attendance"].isin(f_att)
    if f_level: mask &= preds["Level"].isin(f_level)

    view = preds.loc[mask].copy().sort_values("P_AtRisk", ascending=False)
    selected = view.head(capacity).copy()

    show_reasons = st.checkbox("Show reasons (SHAP top-3) for selected", value=False, key="triage_reasons")
    if show_reasons and not selected.empty:
        try:
            import shap

            def _feat_value_label(col: str, v, row) -> str:
                try:
                    iv = int(v)
                except Exception:
                    iv = v
                if col == "GENDER":
                    return f"Gender={'Male' if iv==1 else 'Female'}"
                if col == "STUDENT_STATUS":
                    return f"Status={'Inactive' if iv==1 else 'Active'}"
                if col == "SCHOOL":
                    return f"School={'SoT' if iv==1 else 'SoC'}"
                if col == "ATTENDANCES":
                    return f"Attendance={'Below 80%' if iv==1 else 'Above 80%'}"
                if col == "LOWER_LEVEL":
                    return f"Lower level={'Yes' if iv==1 else 'No'}"
                if col == "FINANCIAL_AID":
                    return f"Financial aid={'No' if iv==1 else 'Yes'}"
                if col == "STUDENT_ORIGIN":
                    return f"Origin={'International' if iv==1 else 'Local'}"
                if col == "Level/Year":
                    mapping = {0: 'Level 1', 1: 'Level 2', 2: 'Level 3'}
                    return f"Level={mapping.get(iv, str(iv))}"
                if col == "AGE":
                    return f"Age≈{_z_to_years(float(v))}"
                if col.startswith("Programme_"):
                    name = col.replace("Programme_","").replace("_"," ")
                    return f"Programme: {name}={'Yes' if iv==1 else 'No'}"
                if isinstance(v, (int, float, np.floating, np.integer)):
                    return f"{_pretty_feat(col)}={float(v):.2f}"
                return f"{_pretty_feat(col)}={v}"

            background = X_test[feature_order]
            explainer, expl_kind = _build_shap_explainer(model, background)

            idxs = selected.index.values
            X_sel = X_test.loc[idxs, feature_order]
            S = _shap_values_for_target(explainer, X_sel, model, label_map, target_name="At-Risk")

            cols = np.array(feature_order)
            reasons_list = []

            for r_ix in range(len(X_sel)):
                row_vals = S[r_ix]
                row_feat = X_sel.iloc[r_ix]

                pos_ix = np.where(row_vals > 0)[0]
                pos_sorted = pos_ix[np.argsort(row_vals[pos_ix])[::-1]]
                chosen = list(pos_sorted[:3])

                if len(chosen) < 3:
                    abs_sorted = np.argsort(np.abs(row_vals))[::-1]
                    for j in abs_sorted:
                        if j not in chosen:
                            chosen.append(j)
                        if len(chosen) == 3:
                            break

                parts = []
                for j in chosen:
                    name = cols[j]
                    val  = row_feat[name]
                    arrow = "↑" if row_vals[j] > 0 else "↓"
                    parts.append(f"{_feat_value_label(name, val, row_feat)} {arrow}{abs(row_vals[j]):.2f}")
                reasons_list.append(" | ".join(parts))

            selected["Reasons"] = reasons_list
            st.caption(f"Reasons computed with SHAP • {expl_kind}  (↑ pushes toward At-Risk; ↓ pushes away)")
        except ModuleNotFoundError:
            st.warning("Install SHAP to enable reasons: `pip install shap==0.45.0`", icon="⚠️")
        except Exception as e:
            st.warning(f"SHAP not available ({type(e).__name__}: {e}); skipping reasons.", icon="⚠️")

    bucket_labels = []
    for p in selected["P_AtRisk"].values:
        label, _ = _bucket(float(p))
        bucket_labels.append(label)
    table = selected.reset_index().rename(columns={"index":"row_id"})
    table["P_AtRisk"] = (table["P_AtRisk"]*100).round(1)
    table["Bucket"] = bucket_labels

    cutoff = float(selected["P_AtRisk"].min()) if len(selected) else 0.0
    st.write(f"**Selected:** {len(selected)}  ·  **Cut-off:** ≥ {(cutoff):.1f}% P(At-Risk)  ·  Thresholds → High ≥ {THRESH['high']*100:.0f}% • Medium ≥ {THRESH['medium']*100:.0f}%")

    colA, colB = st.columns(2)
    with colA:
        st.caption("Mix by Programme")
        st.dataframe(table.groupby("Programme").size().reset_index(name="count"), hide_index=True, use_container_width=True)
    with colB:
        st.caption("Mix by Level")
        st.dataframe(table.groupby("Level").size().reset_index(name="count"), hide_index=True, use_container_width=True)

    show_cols = ["row_id","P_AtRisk","Bucket","predicted","Programme","Level","Attendance"]
    if "Reasons" in table.columns: show_cols.append("Reasons")
    st.dataframe(
        table[show_cols],
        use_container_width=True,
        hide_index=True,
        column_config={
            "P_AtRisk": st.column_config.ProgressColumn(
                "P(At-Risk)",
                help="Higher means greater likelihood of being at-risk",
                min_value=0, max_value=100, format="%.1f%%",
            ),
            "Bucket": st.column_config.TextColumn(width="small"),
        }
    )

    csv_bytes = table[show_cols].to_csv(index=False).encode("utf-8")
    st.download_button("Download Triage (CSV)", data=csv_bytes,
                       file_name="triage_list.csv", mime="text/csv",
                       use_container_width=True, key="triage_csv")
    xbytes = _excel_bytes_table(table[show_cols], sheet_name="triage")
    if xbytes is not None:
        st.download_button("Download Triage (Excel)", data=xbytes,
                           file_name="triage_list.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           use_container_width=True, key="triage_xlsx")
    else:
        st.caption("Install **openpyxl** or **xlsxwriter** to enable Excel export.")

    with st.expander("👥 Split by advisor (balanced)"):
        adv_text = st.text_input("Advisor names (comma-separated)", placeholder="Alice, Bob, Chong, Dina", key="triage_adv_names")
        if adv_text.strip():
            advisors = [a.strip() for a in adv_text.split(",") if a.strip()]
            if advisors:
                split_df = _split_by_advisors(table[show_cols], advisors, sort_col="P_AtRisk")
                st.dataframe(split_df, use_container_width=True, hide_index=True)
                st.download_button("Download split (CSV)", data=split_df.to_csv(index=False).encode("utf-8"),
                                   file_name="triage_split_by_advisor.csv", mime="text/csv",
                                   use_container_width=True, key="triage_split_csv")
                xsplit = _excel_bytes_table(split_df, sheet_name="triage_split")
                if xsplit is not None:
                    st.download_button("Download split (Excel)", data=xsplit,
                                       file_name="triage_split_by_advisor.xlsx",
                                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                       use_container_width=True, key="triage_split_xlsx")
                else:
                    st.caption("Install **openpyxl** or **xlsxwriter** to enable Excel export.")

    with st.expander("✍️ Advising log"):
        col1, col2 = st.columns([1, 3])
        with col1:
            advisor = st.text_input("Advisor", value=os.getenv("USERNAME", "") or "", key="log_advisor")
            review_date = st.date_input(
                "Next review date",
                value=datetime.date.today() + datetime.timedelta(days=14),
                key="log_date",
            )
        with col2:
            notes = st.text_area("Notes (applied to all rows in selection)", key="log_notes")

        # runtime paths
        logs_dir = ROOT_DIR / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        logfile = logs_dir / "advising_log.csv"

        # append current selection to the log
        make_log = st.button("Append to logs/advising_log.csv", type="primary", key="btn_log_append")
        if make_log:
            try:
                log_df = table[show_cols].copy()
                ts = datetime.datetime.now().isoformat(timespec="seconds")
                log_df.insert(0, "timestamp", ts)
                log_df.insert(1, "advisor", advisor)
                log_df.insert(2, "next_review", str(review_date))
                log_df.insert(3, "notes", notes)

                if logfile.exists():
                    old = pd.read_csv(logfile)
                    all_df = pd.concat([old, log_df], ignore_index=True)
                else:
                    all_df = log_df

                all_df.to_csv(logfile, index=False)

                # optional: persist a timestamped copy for audit under /exports
                exp_dir = ROOT_DIR / "exports"
                exp_dir.mkdir(parents=True, exist_ok=True)
                stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                all_df.tail(len(log_df)).to_csv(exp_dir / f"advising_selection_{stamp}.csv", index=False)

                st.success(f"Appended {len(log_df)} rows to {logfile.relative_to(ROOT_DIR)}")
            except Exception as e:
                st.error(f"Failed to write log: {e}")

        st.markdown("---")

        # viewer + downloads (works on local and on Streamlit Cloud)
        if logfile.exists():
            try:
                df_all = pd.read_csv(logfile)
                st.caption(f"Current advising log — {len(df_all)} rows")
                st.dataframe(df_all.tail(50), use_container_width=True, hide_index=True)

                c1, c2 = st.columns(2)
                with c1:
                    st.download_button(
                        "Download advising_log (CSV)",
                        df_all.to_csv(index=False).encode("utf-8"),
                        file_name="advising_log.csv",
                        mime="text/csv",
                        use_container_width=True,
                        key="dl_log_csv",
                    )
                with c2:
                    st.download_button(
                        "Download advising_log (Excel)",
                        _excel_bytes_df(df_all),
                        file_name="advising_log.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True,
                        key="dl_log_xlsx",
                    )
            except Exception as e:
                st.warning(f"Could not display/download log: {e}")
        else:
            st.caption("No advising log found yet.")

# ──────────────────────────────────────────────────────────────────────────────
# Scenario Compare helpers (preset queue)
def _pending_bucket_key(base_idx: int) -> str:
    return f"__pending_presets__{base_idx}"

def _queue_preset_value(target_key: str, value: str, base_idx: int) -> None:
    bucket = st.session_state.get(_pending_bucket_key(base_idx), {})
    bucket[target_key] = value
    st.session_state[_pending_bucket_key(base_idx)] = bucket
    st.rerun()

def _queue_multi_presets(kv: dict[str, str], base_idx: int) -> None:
    bucket = st.session_state.get(_pending_bucket_key(base_idx), {})
    bucket.update(kv)
    st.session_state[_pending_bucket_key(base_idx)] = bucket
    st.rerun()

def _apply_pending_presets_if_any(base_idx: int) -> None:
    bucket_key = _pending_bucket_key(base_idx)
    bucket = st.session_state.pop(bucket_key, None)
    if not bucket:
        return
    for k, v in bucket.items():
        st.session_state[k] = v

# ✅ Helper: first render uses default index, afterwards rely purely on session_state
def _selectbox_ss(label: str, options: list[str], key: str, default_value: str):
    if key in st.session_state:
        return st.selectbox(label, options, key=key)
    try:
        default_idx = options.index(default_value)
    except ValueError:
        default_idx = 0
    return st.selectbox(label, options, index=default_idx, key=key)

# ──────────────────────────────────────────────────────────────────────────────
# SCENARIO COMPARE
with tab_compare:
    st.caption("Clone a student, tweak inputs, and compare risk & plan. Gemini-first; fallback only if Gemini fails.")

    base_idx = st.selectbox(
        "Select base student for scenario",
        options=list(range(len(X_test))),
        index=0,
        format_func=lambda x: f"Student {x}",
        key="sc_base_selectbox"
    )
    base_row = X_test.iloc[base_idx].copy()


    # Keys per base row
    att_key   = f"sc_att_{base_idx}"
    level_key = f"sc_level_{base_idx}"
    fin_key   = f"sc_fin_{base_idx}"
    low_key   = f"sc_lower_{base_idx}"
    prog_key  = f"sc_prog_{base_idx}"
    age_key   = f"sc_age_{base_idx}"

    # Apply any queued presets before widgets render
    _apply_pending_presets_if_any(base_idx)

    # Base labels
    att_label  = "Above 80%" if int(base_row.get("ATTENDANCES", 0)) == 0 else "Below 80%"
    level_rev  = {0: "Level 1", 1: "Level 2", 2: "Level 3"}
    cur_level  = level_rev.get(int(base_row.get("Level/Year", 0)), "Level 1")
    fin_label  = "Yes" if int(base_row.get("FINANCIAL_AID", 0)) == 0 else "No"
    low_label  = "Yes" if int(base_row.get("LOWER_LEVEL", 0)) == 1 else "No"
    cur_age_yr = _z_to_years(base_row.get("AGE", 0.0))

    prog_cols   = [c for c in feature_order if c.startswith("Programme_")]
    prog_names  = [c.replace("Programme_", "").replace("_", " ") for c in prog_cols]
    sorted_prog = sorted(prog_names)
    cur_prog    = "Unknown"
    if prog_cols:
        j = int(np.argmax(base_row[prog_cols].values))
        cur_prog = prog_names[j] if base_row[prog_cols[j]] == 1 else "Unknown"

    # Controls row (using session-state-aware selectbox)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        _selectbox_ss("Attendance (scenario)", ["Above 80%","Below 80%"], att_key, att_label)
    with c2:
        _selectbox_ss("Level/Year (scenario)", ["Level 1","Level 2","Level 3"], level_key, cur_level)
    with c3:
        _selectbox_ss("Financial Aid (scenario)", ["Yes","No"], fin_key, fin_label)
    with c4:
        _selectbox_ss("Lower Level (scenario)", ["No","Yes"], low_key, low_label)

    c5, c6 = st.columns([2, 2])
    with c5:
        _selectbox_ss("Programme (scenario)", sorted_prog if sorted_prog else ["Unknown"], prog_key, cur_prog)
    with c6:
        age_choices = [str(i) for i in range(16, 61)]
        _selectbox_ss("Age (years, scenario)", age_choices, age_key, str(cur_age_yr))

    # Current values (after possible rerun from queued presets)
    now_att  = st.session_state.get(att_key,  att_label)
    now_lvl  = st.session_state.get(level_key, cur_level)
    now_fin  = st.session_state.get(fin_key,  fin_label)
    now_low  = st.session_state.get(low_key,  low_label)
    now_prog = st.session_state.get(prog_key, cur_prog)
    now_age  = str(st.session_state.get(age_key, cur_age_yr))

    # Aligned quick actions under each select
    b1, b2, b3, b4 = st.columns(4)

    with b1:
        st.caption(f"Attendance: **{now_att}**")
        next_att = "Below 80%" if now_att == "Above 80%" else "Above 80%"
        if st.button(f"Toggle attendance ⇒ {next_att}", use_container_width=True, key=f"toggle_att_{base_idx}"):
            _queue_preset_value(att_key, next_att, base_idx)

    with b2:
        st.caption(f"Level: **{now_lvl}**")
        levels = ["Level 1", "Level 2", "Level 3"]
        if now_lvl not in levels:
            now_lvl = cur_level
        next_lvl = levels[(levels.index(now_lvl) + 1) % 3]
        if st.button(f"Cycle level → {next_lvl}", use_container_width=True, key=f"cycle_lvl_{base_idx}"):
            _queue_preset_value(level_key, next_lvl, base_idx)

    with b3:
        st.caption(f"Financial aid: **{now_fin}**")
        next_fin = "No" if now_fin == "Yes" else "Yes"
        if st.button(f"Toggle financial aid ⇒ {next_fin}", use_container_width=True, key=f"toggle_fin_{base_idx}"):
            _queue_preset_value(fin_key, next_fin, base_idx)

    with b4:
        st.caption(f"Lower level: **{now_low}**")
        next_low = "No" if now_low == "Yes" else "Yes"
        if st.button(f"Toggle lower-level ⇒ {next_low}", use_container_width=True, key=f"toggle_low_{base_idx}"):
            _queue_preset_value(low_key, next_low, base_idx)
        if st.button("Reset to base", use_container_width=True, key=f"reset_base_{base_idx}"):
            _queue_multi_presets({
                att_key:   att_label,
                level_key: cur_level,
                fin_key:   fin_label,
                low_key:   low_label,
                prog_key:  cur_prog,
                age_key:   str(cur_age_yr),
            }, base_idx)

    chips = [
        ("Attendance", now_att, att_label),
        ("Level", now_lvl, cur_level),
        ("Financial aid", now_fin, fin_label),
        ("Lower level", now_low, low_label),
        ("Programme", now_prog, cur_prog),
        ("Age", now_age, str(cur_age_yr)),
    ]
    pill_html = "".join(
        f"<span class='pill'>{html.escape(lbl)}: {html.escape(cur)}"
        f"{(' <span style=\"opacity:.6\">(base: '+html.escape(b)+')</span>' if cur!=b else '')}</span>"
        for (lbl, cur, b) in chips
    )
    st.markdown(
        f"<div class='pills' style='margin-top:8px; margin-bottom:22px'>{pill_html}</div>",
        unsafe_allow_html=True
    )
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)  # spacer

    playbook_text = (
        "Quick preset bundle: sets **Attendance ≥ 80%**, **Financial aid = Yes**, "
        "and **Lower level = No**. Leaves Programme, Level/Year, and Age unchanged."
    )
    clicked_playbook = st.button(
        "Apply playbook: Stabilize & Support",
        use_container_width=True,
        key=f"playbook_stab_main_{base_idx}",
        help="Sets Attendance ≥80%, Financial aid = Yes, Lower level = No"
    )
    st.caption(playbook_text)

    if clicked_playbook:
        _queue_multi_presets({
            att_key: "Above 80%",
            fin_key: "Yes",
            low_key: "No",
        }, base_idx)

    if st.button("Run compare", type="primary", use_container_width=True, key=f"btn_run_compare_{base_idx}"):
        try:
            now_att  = st.session_state.get(att_key,  att_label)
            now_lvl  = st.session_state.get(level_key, cur_level)
            now_fin  = st.session_state.get(fin_key,  fin_label)
            now_low  = st.session_state.get(low_key,  low_label)
            now_prog = st.session_state.get(prog_key, cur_prog)
            now_age  = str(st.session_state.get(age_key, cur_age_yr))

            X_one_base = X_test.iloc[[base_idx]]
            out_base = _prefer_gemini_row(X_one_base, X_test, model, label_map, docx, tone)
            st.session_state["last_llm_source"] = out_base.get("llm_source", "fallback")

            sc_row = base_row.copy()
            sc_row["ATTENDANCES"]   = 0 if now_att == "Above 80%" else 1
            sc_row["FINANCIAL_AID"] = 0 if now_fin == "Yes" else 1
            sc_row["LOWER_LEVEL"]   = 1 if now_low == "Yes" else 0
            level_fwd = {"Level 1":0,"Level 2":1,"Level 3":2}
            if "Level/Year" in sc_row.index:
                sc_row["Level/Year"] = level_fwd.get(now_lvl, 0)
            if "AGE" in sc_row.index:
                sc_row["AGE"] = _years_to_z(int(now_age))
            for c in prog_cols: sc_row[c] = 0
            if prog_cols and now_prog in prog_names:
                sc_row[prog_cols[prog_names.index(now_prog)]] = 1

            X_one = pd.DataFrame([sc_row.values], columns=X_test.columns)
            out_scn = _prefer_gemini_row(X_one, X_test, model, label_map, docx, tone)
            st.session_state["last_llm_source"] = out_scn.get("llm_source", "fallback")

            if FORCE_GEMINI:
                if str(out_base.get("llm_source","")).lower() != "gemini":
                    out_base = _prefer_gemini_row(X_one_base, X_test, model, label_map, docx, tone, tries=3)
                if str(out_scn.get("llm_source","")).lower() != "gemini":
                    out_scn  = _prefer_gemini_row(X_one, X_test, model, label_map, docx, tone, tries=3)
                if (str(out_base.get("llm_source","")).lower() != "gemini" or
                    str(out_scn.get("llm_source","")).lower() != "gemini"):
                    st.info("Gemini key present but one call used backup (quota/network).", icon="⚠️")
        except Exception as e:
            st.error(f"Plan generation failed: {e}", icon="🚫")
            st.stop()

        colL, colR = st.columns(2)
        with colL:
            st.markdown("### Base")
            _banner(out_base["prediction"]["pred_label"], out_base["prediction"]["proba_dict"])
            _risk_gauge(out_base["prediction"]["proba_dict"].get("At-Risk", 0.0))
            _bar(out_base["prediction"]["proba_dict"])
            cA, cB = st.columns(2)
            with cA: _feature_pills("SHAP (top-3)", out_base.get("shap_top", []), "shap")
            with cB: _feature_pills("LIME (top-3)", out_base.get("lime_top", []), "lime")
            _render_plan(out_base["plan"], tone, out_base.get("llm_source","fallback"), dl_key=f"dl-base-{out_base['run_id']}")
        with colR:
            st.markdown("### Scenario")
            _banner(out_scn["prediction"]["pred_label"], out_scn["prediction"]["proba_dict"])
            _risk_gauge(out_scn["prediction"]["proba_dict"].get("At-Risk", 0.0))
            _bar(out_scn["prediction"]["proba_dict"])
            cA, cB = st.columns(2)
            with cA: _feature_pills("SHAP (top-3)", out_scn.get("shap_top", []), "shap")
            with cB: _feature_pills("LIME (top-3)", out_scn.get("lime_top", []), "lime")
            _render_plan(out_scn["plan"], tone, out_scn.get("llm_source","fallback"), dl_key=f"dl-scn-{out_scn['run_id']}")

        p_base = float(out_base["prediction"]["proba_dict"].get("At-Risk", 0.0))
        p_scn  = float(out_scn["prediction"]["proba_dict"].get("At-Risk", 0.0))
        delta  = (p_scn - p_base) * 100.0
        sign   = "↓" if delta < 0 else ("↑" if delta > 0 else "→")
        st.markdown(f"**Δ P(At-Risk): {sign} {abs(delta):.1f} pp**  (Base {p_base*100:.1f}% → Scenario {p_scn*100:.1f}%)")

        if delta <= -5.0:
            st.success("Nice! Scenario reduces At-Risk probability by at least 5 pp.", icon="🎉")

