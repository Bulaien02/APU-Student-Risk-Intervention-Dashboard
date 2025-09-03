# path: notebooks/intervention_core.py — RAG + Gemini + explainers
from __future__ import annotations

import os, re, json, unicodedata, textwrap, warnings, uuid, datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Literal, Tuple

import numpy as np
import pandas as pd
import joblib

# Explainability
import shap
import lime.lime_tabular as llt

# RAG
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Schema / validation (Pydantic v2)
from pydantic import BaseModel, Field, HttpUrl, model_validator

# ───────────────────────── Paths (robust from any working dir) ─────────────────────────
FILE_DIR = Path(__file__).resolve().parent           # e.g., .../notebooks
ROOT_DIR = FILE_DIR.parent                           # repo root (one level up)

# Try both root and notebooks for every folder
CANDIDATE_BASES = [ROOT_DIR, FILE_DIR]

def _first_existing(*paths: Path) -> Optional[Path]:
    for p in paths:
        if p and p.exists():
            return p
    return None

def _need(path: Optional[Path], label: str) -> Path:
    assert path is not None and path.exists(), f"Missing artifact: {label}"
    return path

# Prefer notebooks/<dir> first, then root/<dir>
ART_DIR    = _first_existing(FILE_DIR / "artifacts_models", ROOT_DIR / "artifacts_models") or (FILE_DIR / "artifacts_models")
SPLIT_DIR  = _first_existing(FILE_DIR / "artifacts_split",  ROOT_DIR / "artifacts_split")  or (FILE_DIR / "artifacts_split")
ASSETS_DIR = _first_existing(FILE_DIR / "assets",          ROOT_DIR / "assets")          or (FILE_DIR / "assets")

# --- pick docs folder (prefer notebooks/docs, allow env override) ---
DOCS_DIR = None
if os.getenv("DOCS_DIR"):
    DOCS_DIR = Path(os.getenv("DOCS_DIR")).resolve()
else:
    # prefer notebooks/docs, then repo-root/docs, else fall back to notebooks/docs
    DOCS_DIR = _first_existing(FILE_DIR / "docs", ROOT_DIR / "docs") or (FILE_DIR / "docs")

LOGS_DIR   = _first_existing(FILE_DIR / "logs",            ROOT_DIR / "logs")            or (FILE_DIR / "logs")
LOGS_DIR.mkdir(exist_ok=True, parents=True)

# ───────────────────────── Encodings (match your preprocessing) ─────────────────────────
GENDER_MAP        = {"Male": 1, "Female": 0}
STATUS_MAP        = {"Active": 0, "Inactive": 1}
SCHOOL_MAP        = {"School of Computing (SoC)": 0, "School of Technology (SoT)": 1}
YESNO_MAP_01      = {"Yes": 0, "No": 1}                 # progression/graduation on time
ATTEND_MAP        = {"Above 80%": 0, "Below 80%": 1}
LOWERLEVEL_MAP    = {"No": 0, "Yes": 1}
FINANCIAL_AID_MAP = {"Yes": 0, "No": 1}
ORIGIN_MAP        = {"Local": 0, "International": 1}
LEVELYEAR_MAP     = {"Level 1": 0, "Level 2": 1, "Level 3": 2}
ENTRY_QUAL_MAP    = {
    "Certificate": 0, "Foundation": 1, "Diploma": 2, "Year 1": 3,
    "UEC": 4, "STPM": 5, "Pre-U": 6, "High School": 7, "Degree": 8
}
ENTRY_QUAL_REV    = {v: k for k, v in ENTRY_QUAL_MAP.items()}

# ───────────────────────── Artifacts ─────────────────────────
def load_artifacts() -> tuple[pd.DataFrame, Any, Dict[int,str], List[str]]:
    """Load model + schema-critical files with clear errors if missing."""
    model_path    = _first_existing(
        ROOT_DIR / "artifacts_models" / "xgb_tuned.joblib",
        FILE_DIR / "artifacts_models" / "xgb_tuned.joblib",
    )
    feat_path     = _first_existing(
        ROOT_DIR / "artifacts_models" / "feature_order.json",
        FILE_DIR / "artifacts_models" / "feature_order.json",
    )
    labelmap_path = _first_existing(
        ROOT_DIR / "artifacts_models" / "label_map.json",
        FILE_DIR / "artifacts_models" / "label_map.json",
    )
    xtest_path    = _first_existing(
        ROOT_DIR / "artifacts_split" / "X_test.csv",
        FILE_DIR / "artifacts_split" / "X_test.csv",
    )

    model_path    = _need(model_path,    "artifacts_models/xgb_tuned.joblib")
    feat_path     = _need(feat_path,     "artifacts_models/feature_order.json")
    labelmap_path = _need(labelmap_path, "artifacts_models/label_map.json")
    xtest_path    = _need(xtest_path,    "artifacts_split/X_test.csv")

    feature_order: List[str]     = json.loads(feat_path.read_text(encoding="utf-8"))
    label_map_raw: Dict[str,str] = json.loads(labelmap_path.read_text(encoding="utf-8"))
    label_map: Dict[int,str]     = {int(k): v for k, v in label_map_raw.items()}

    X_test = pd.read_csv(xtest_path)
    missing = [c for c in feature_order if c not in X_test.columns]
    extra   = [c for c in X_test.columns if c not in feature_order]
    assert not missing and not extra, f"Column mismatch. Missing: {missing[:6]} | Extra: {extra[:6]}"
    X_test = X_test[feature_order]

    model = joblib.load(model_path)
    try:
        booster = model.get_booster()
        booster.set_param({"device": "cpu", "tree_method": "hist"})  # why: avoid GPU requirement
    except Exception:
        pass

    print("Loaded artifacts from:")
    print(f"  model: {model_path}")
    print(f"  features: {feat_path}")
    print(f"  labels: {labelmap_path}")
    print(f"  X_test: {xtest_path}")

    return X_test, model, label_map, feature_order

# ───────────────────────── Programme helpers ─────────────────────────
def programme_columns(feature_order: List[str]) -> List[str]:
    return [c for c in feature_order if c.startswith("Programme_")]

def programme_display_names(feature_order: List[str]) -> List[str]:
    return [c.replace("Programme_", "").replace("_", " ") for c in programme_columns(feature_order)]

def programme_col_for_name(feature_order: List[str], disp_name: str) -> str:
    for c in programme_columns(feature_order):
        if c.replace("Programme_", "").replace("_", " ") == disp_name:
            return c
    return programme_columns(feature_order)[0]

# ───────────────────────── SHAP & LIME ─────────────────────────
def shap_top_k(model, row_df: pd.DataFrame, background: pd.DataFrame, k: int = 3) -> List[str]:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(row_df)
    pred_class  = int(model.predict(row_df)[0])
    class_ix    = list(model.classes_).index(pred_class)

    if isinstance(shap_values, list):
        vals = np.array(shap_values[class_ix][0])
    else:
        v = np.array(shap_values[0])
        vals = v if v.ndim == 1 else v[:, class_ix]

    order = np.argsort(np.abs(vals))[::-1]
    seen, out = set(), []
    for ix in order:
        f = row_df.columns[int(ix)]
        if f not in seen:
            out.append(f); seen.add(f)
        if len(out) == k: break
    return out

def lime_top_k(model, X_ref: pd.DataFrame, row_df: pd.DataFrame, k: int = 3) -> List[str]:
    explainer = llt.LimeTabularExplainer(
        training_data=X_ref.values,
        feature_names=X_ref.columns.tolist(),
        class_names=[str(c) for c in model.classes_],
        discretize_continuous=True,
        mode="classification",
    )
    inst = explainer.explain_instance(
        data_row=row_df.values[0],
        predict_fn=model.predict_proba,
        num_features=max(8, k),
    )
    cols = set(X_ref.columns)
    seen, out = set(), []
    for feat_str, _ in inst.as_list():
        matches = [c for c in cols if c in feat_str]
        feat = max(matches, key=len) if matches else re.findall(r"[A-Za-z0-9_./() -]+", feat_str)[0]
        if feat not in seen:
            out.append(feat); seen.add(feat)
        if len(out) == k: break
    return out

def humanize_feature(row: pd.Series, feature: str) -> str:
    if feature == "GENDER":              return f"Gender: {'Male' if row['GENDER'] == 1 else 'Female'}"
    if feature == "STUDENT_STATUS":      return f"Student Status: {'Inactive' if row['STUDENT_STATUS'] == 1 else 'Active'}"
    if feature == "SCHOOL":              return f"School Code: {row['SCHOOL']}"
    if feature == "ENTRY_QUALIFICATION": return f"Entry Qualification: {ENTRY_QUAL_REV.get(row['ENTRY_QUALIFICATION'], row['ENTRY_QUALIFICATION'])}"
    if feature == "PROGRESSION_ON_TIME": return f"Progression On Time: {'No' if row['PROGRESSION_ON_TIME'] == 1 else 'Yes'}"
    if feature == "GRADUATION_ON_TIME":  return f"Graduation On Time: {'No' if row['GRADUATION_ON_TIME'] == 1 else 'Yes'}"
    if feature == "Level/Year":          return f"Level/Year: {['Level 1','Level 2','Level 3'][int(row['Level/Year'])]}"
    if feature == "ATTENDANCES":         return f"Attendances: {'Below 80%' if row['ATTENDANCES'] == 1 else 'Above 80%'}"
    if feature == "LOWER_LEVEL":         return f"Lower Level: {'Yes' if row['LOWER_LEVEL'] == 1 else 'No'}"
    if feature == "FINANCIAL_AID":       return f"Financial Aid: {'No' if row['FINANCIAL_AID'] == 1 else 'Yes'}"
    if feature == "STUDENT_ORIGIN":      return f"Student Origin: {'International' if row['STUDENT_ORIGIN'] == 1 else 'Local'}"
    if feature.startswith("Programme_"): return f"Programme: {feature.replace('Programme_', '').replace('_',' ')}"
    if feature == "AGE":                 return f"Age (z-score): {row['AGE']:.0f}"
    if feature == "CGPA":                return "CGPA (normalized)"
    return feature

def predict_one(model, row_df: pd.DataFrame, label_map: Dict[int,str]) -> Dict[str, Any]:
    pred_idx = int(model.predict(row_df)[0])
    vec      = model.predict_proba(row_df)[0]
    return {
        "pred_idx": pred_idx,
        "pred_label": label_map[pred_idx],
        "proba_dict": {label_map[int(c)]: float(vec[i]) for i, c in enumerate(model.classes_)},
    }

# ───────────────────────── RAG (DocIndex + helpers) ─────────────────────────
def _norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", s)).strip()

def _chunk(text: str, max_chars: int = 1200, overlap: int = 200) -> List[str]:
    t = _norm_ws(text)
    if len(t) <= max_chars: return [t]
    out, i = [], 0
    while i < len(t):
        j = min(len(t), i + max_chars)
        out.append(t[i:j])
        if j == len(t): break
        i = max(0, j - overlap)
    return out

def _read_txt(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="ignore")

def _read_pdf(p: Path) -> str:
    import pypdf
    reader = pypdf.PdfReader(str(p))
    pages = []
    for pg in reader.pages:
        try: pages.append(pg.extract_text() or "")
        except Exception: pages.append("")
    return "\n".join(pages)

def _fetch_url(url: str, timeout: int = 20) -> str:
    r = requests.get(url, timeout=timeout, headers={"User-Agent":"Mozilla/5.0"})
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    for t in soup(["script","style","noscript","header","footer","nav","form"]): t.decompose()
    return " ".join(x.get_text(" ", strip=True) for x in soup.find_all(["h1","h2","h3","p","li","td"]))

@dataclass
class DocChunk:
    source_id: str
    title: str
    category: str
    text: str
    idx: int

class DocIndex:
    def __init__(self):
        self.chunks: List[DocChunk] = []
        self._vec: Optional[TfidfVectorizer] = None
        self._mat = None

    def add_file(self, path: str|Path, *, source_id: str, title: str, category: str="policy",
                 max_chars: int=1200, overlap: int=200) -> "DocIndex":
        p = Path(path)
        if not p.exists():
            warnings.warn(f"Missing file: {p}")
            return self
        raw = _read_pdf(p) if p.suffix.lower()==".pdf" else _read_txt(p)
        for c in _chunk(raw, max_chars, overlap):
            self.chunks.append(DocChunk(source_id, title, category, c, len(self.chunks)))
        return self

    def add_url(self, url: str, *, source_id: str, title: str, category: str="support",
                max_chars: int=1200, overlap: int=200, timeout: int=20) -> "DocIndex":
        try:
            raw = _fetch_url(url, timeout=timeout)
            for c in _chunk(raw, max_chars, overlap):
                self.chunks.append(DocChunk(source_id, title, category, c, len(self.chunks)))
        except Exception as e:
            warnings.warn(f"Failed to fetch {url}: {e}")
        return self

    def build(self, max_features: int = 40000) -> "DocIndex":
        assert self.chunks, "No documents added."
        corpus = [c.text for c in self.chunks]
        self._vec = TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1,2), max_features=max_features)
        self._mat = self._vec.fit_transform(corpus)
        return self

    def search(self, query: str, k: int = 5) -> List[DocChunk]:
        assert self._vec is not None and self._mat is not None, "Index not built."
        qv = self._vec.transform([query])
        sims = linear_kernel(qv, self._mat).ravel()
        idx = sims.argsort()[::-1][:k]
        return [self.chunks[i] for i in idx]

def _retrieval_query(class_label: str, shap_top: List[str], lime_top: List[str], proba: Dict[str,float]) -> str:
    facets = ", ".join(dict.fromkeys(shap_top + lime_top)) or "academic support"
    riskiest = sorted(proba.items(), key=lambda kv: kv[1], reverse=True)[0][0] if proba else class_label
    return _norm_ws(f"{class_label} policy support {facets} attendance probation appeal resit {riskiest}")

def stringify_snippets(snips: List[DocChunk], limit: int = 500) -> str:
    return "\n".join(f"[{s.source_id} | {s.title}] {s.text[:limit]}" for s in snips)

def retrieve_policy_ctx(index: "DocIndex", class_label: str, shap_top: List[str], lime_top: List[str],
                        proba: Dict[str,float], k: int=5) -> str:
    return stringify_snippets(index.search(_retrieval_query(class_label, shap_top, lime_top, proba), k=k))

# ───────────────────────── docindex helpers (create/audit/load) ─────────────────────────
def ensure_docindex_template(path: str|Path) -> None:
    p = Path(path)
    if p.exists(): return
    p.parent.mkdir(parents=True, exist_ok=True)
    template = {
        "sources":[
            {"id":"apu_handbook_2023","title":"APU Student Handbook (Jul 2023)","type":"pdf",
             "path_or_url": str(DOCS_DIR / "APU Student Handbook-Jul 2023.pdf"), "category":"policy"},
            {"id":"apu_student_services","title":"APU Student Services","type":"url",
             "path_or_url":"https://www.apu.edu.my/index.php/student-services","category":"support"}
        ]
    }
    p.write_text(json.dumps(template, indent=2), encoding="utf-8")

def _check_url(url: str, timeout: int=8) -> Tuple[bool,int]:
    try:
        r = requests.head(url, timeout=timeout, allow_redirects=True)
        if r.status_code >= 400:
            r = requests.get(url, timeout=timeout, allow_redirects=True)
        return 200 <= r.status_code < 400, r.status_code
    except Exception:
        return False, -1

def audit_docindex(path: str|Path, strict: bool=False) -> Dict[str,Any]:
    p = Path(path); assert p.exists(), f"Missing config: {p}"
    data = json.loads(p.read_text(encoding="utf-8"))
    issues, report = [], []
    for src in data.get("sources", []):
        _id = str(src.get("id") or "").strip()
        typ = str(src.get("type") or "").strip().lower()
        tgt = str(src.get("path_or_url") or "").strip()

        ok, detail = True, ""
        if typ in {"pdf", "txt"}:
            pth = Path(tgt)
            if not pth.is_absolute():
                # 👇 use DOCS_DIR for relative paths
                pth = (DOCS_DIR / tgt) if DOCS_DIR else pth
            ok = pth.exists()
            detail = "exists" if ok else "missing"
            target_for_report = str(pth)
        elif typ == "url":
            ok, code = _check_url(tgt)
            detail = f"http {code}" if code != -1 else "unreachable"
            target_for_report = tgt
        else:
            ok, detail = False, f"unknown type '{typ}'"
            target_for_report = tgt

        report.append({"id": _id, "type": typ, "target": target_for_report, "ok": ok, "detail": detail})
        if not ok:
            issues.append(f"[{_id}] {typ}: {detail} -> {target_for_report}")

    print("DocIndex audit:")
    for r in report:
        print(f"  {'OK ' if r['ok'] else 'FAIL'}  id={r['id']:<22} type={r['type']:<3} {r['detail']:<12} {r['target']}")
    if strict and issues:
        raise AssertionError("docindex.json issues:\n  - " + "\n  - ".join(issues))
    return {"ok": not issues, "report": report}

def load_docindex(path: str|Path) -> "DocIndex":
    p = Path(path); assert p.exists(), f"Missing: {p}"
    data = json.loads(p.read_text(encoding="utf-8"))
    idx = DocIndex()
    for s in data.get("sources", []):
        _id  = s["id"]
        ttl  = s.get("title", _id)
        typ  = s["type"].lower()
        tgt  = s["path_or_url"]
        cat  = s.get("category", "policy")

        try:
            if typ in {"pdf", "txt"}:
                pth = Path(tgt)
                if not pth.is_absolute():
                    pth = (DOCS_DIR / tgt) if DOCS_DIR else pth
                idx.add_file(str(pth), source_id=_id, title=ttl, category=cat)
            elif typ == "url":
                idx.add_url(tgt, source_id=_id, title=ttl, category=cat)
        except Exception as e:
            warnings.warn(f"Failed to add source {s}: {e}")


# ───────────────────────── Gemini + schema ─────────────────────────
RiskLevel = Literal["low","medium","high"]
Tone = Literal["supportive","coachy","concise"]

class Resource(BaseModel):
    title: str = Field(min_length=2, max_length=120)
    url: Optional[HttpUrl] = None

class InterventionPlan(BaseModel):
    risk_level: RiskLevel
    why_model_thinks_so: str = Field(min_length=10, max_length=300)
    study_plan: List[str] = Field(min_length=1, max_length=10)
    resources: List[Resource] = Field(min_length=0, max_length=10)
    follow_up_actions: List[str] = Field(min_length=1, max_length=5)
    tone: Tone
    citations: List[str] = Field(default_factory=list)
    @model_validator(mode="after")
    def _trim(self):
        self.study_plan = [s.strip()[:200] for s in self.study_plan]
        self.follow_up_actions = [s.strip()[:160] for s in self.follow_up_actions]
        seen, out = set(), []
        for c in self.citations:
            if c not in seen:
                out.append(c); seen.add(c)
        self.citations = out[:10]
        return self

_CIT_TAG_PATTERN = re.compile(r"\[(?P<sid>[^|\]]+)\s*\|\s*(?P<title>[^\]]+)\]")

def extract_citations(policy_ctx: str) -> List[str]:
    tags = [f"{m.group('sid').strip()} | {m.group('title').strip()}" for m in _CIT_TAG_PATTERN.finditer(policy_ctx or "")]
    seen, out = set(), []
    for t in tags:
        if t not in seen:
            out.append(t); seen.add(t)
    return out[:10]

class GeminiClient:
    """Returns plan + source flag so UI can show 'Gemini' vs 'fallback'."""
    def __init__(self, api_key: Optional[str]=None, model_name: str|None=None):
        # accept GEMINI_API_KEY or GOOGLE_API_KEY
        key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.enabled = bool(key)
        self.model = None
        self.cfg = {"temperature":0.0, "top_p":1.0, "top_k":1, "max_output_tokens":1024}
        if self.enabled:
            try:
                import google.generativeai as genai
                genai.configure(api_key=key)
                self.model = genai.GenerativeModel(model_name or os.getenv("GEMINI_MODEL","gemini-2.0-flash"))
            except Exception:
                self.enabled = False
                self.model = None

    @staticmethod
    def _extract_json(text: str) -> str:
        m = re.search(r"\{.*\}\s*$", text.strip(), re.DOTALL)
        if m: return m.group(0)
        i, j = text.find("{"), text.rfind("}")
        return text[i:j+1] if i!=-1 and j!=-1 else ""

    def generate_plan(self, prompt: str) -> tuple[InterventionPlan, str]:
        if self.enabled and self.model is not None:
            try:
                resp = self.model.generate_content(prompt, generation_config=self.cfg)
                raw = (getattr(resp, "text", "") or "").strip()
                js  = self._extract_json(raw)
                plan = InterventionPlan.model_validate_json(js)
                return plan, "gemini"
            except Exception:
                pass
        # Fallback
        return InterventionPlan(
            risk_level="medium",
            why_model_thinks_so="Top contributing factors signal focus areas this week.",
            study_plan=["Attend study skills workshop","Create weekly timetable","Meet program advisor"],
            resources=[], follow_up_actions=["Progress check in 2 weeks"], tone="supportive",
            citations=[],
        ), "fallback"

SYSTEM_RULES = """
Generate a one-shot JSON intervention plan for a university student.
ALWAYS return only JSON matching the schema. No markdown, no extra text.
Use APU policies/resources when relevant; tailor to Malaysian university context.
Interventions must be provided for all classes (Excellent, Average, At-Risk).
""".strip()

def build_prompt(row: pd.Series, class_label: str, proba: Dict[str,float],
                 shap_feats: List[str], lime_feats: List[str], policy_ctx: str,
                 tone: Tone="supportive") -> str:
    factors = [humanize_feature(row, f) for f in dict.fromkeys(shap_feats + lime_feats)]
    proba_line = ", ".join(f"{k}={proba[k]:.2f}" for k in sorted(proba))
    schema_hint = """
JSON schema:
  risk_level: "low" | "medium" | "high"
  why_model_thinks_so: <= 2 lines
  study_plan: 3–7 short items
  resources: [{title, url?}]
  follow_up_actions: 1–3 items
  tone: "supportive" | "coachy" | "concise"
""".strip()
    doc_snip = (policy_ctx or "")[:3600]
    return textwrap.dedent(f"""
    {SYSTEM_RULES}
    {schema_hint}

    CONTEXT:
    - Student class: {class_label}
    - Class probabilities: {proba_line}
    - Top contributing factors (SHAP/LIME): {", ".join(factors) if factors else "n/a"}
    - Policy & resources (APU handbook / articles):
      {doc_snip}

    CONSTRAINTS:
    - Use the policy context when applicable.
    - Output only the JSON object; no code fences or extra text.
    - tone="{tone}".

    OUTPUT:
    {{
      "risk_level": "...",
      "why_model_thinks_so": "...",
      "study_plan": ["...", "..."],
      "resources": [{{"title":"...","url":"..."}}],
      "follow_up_actions": ["..."],
      "tone": "supportive",
      "citations": ["source_id | title"]
    }}
    """).strip()

def _risk_from_prob(p_at_risk: float) -> str:
    return "low" if p_at_risk < 0.33 else ("medium" if p_at_risk < 0.66 else "high")

# ───────────────────────── Generators ─────────────────────────
def generate_intervention_for_row(row_df: pd.DataFrame, X_ref: pd.DataFrame, model, label_map: Dict[int,str],
                                  doc_index: Optional[DocIndex], tone: Tone="supportive",
                                  k_policy: int=5, k_features: int=3) -> Dict[str,Any]:
    row = row_df.iloc[0]
    pred = predict_one(model, row_df, label_map)
    bg   = X_ref.sample(min(100, len(X_ref)), random_state=42)
    shap_feats = shap_top_k(model, row_df, bg, k=k_features)
    lime_feats = lime_top_k(model, X_ref, row_df, k=k_features)
    policy_ctx = retrieve_policy_ctx(doc_index, pred["pred_label"], shap_feats, lime_feats, pred["proba_dict"], k=k_policy) if doc_index else ""
    citations  = extract_citations(policy_ctx)
    prompt     = build_prompt(row, pred["pred_label"], pred["proba_dict"], shap_feats, lime_feats, policy_ctx, tone)
    client     = GeminiClient()
    plan, source = client.generate_plan(prompt)
    plan.risk_level = _risk_from_prob(float(pred["proba_dict"].get("At-Risk", 0.0)))
    if not plan.citations and citations:
        plan.citations = citations
    plan_dict = plan.model_dump(mode="json")

    run_id = dt.datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + str(uuid.uuid4())[:8]
    (LOGS_DIR / f"{run_id}-prompt.txt").write_text(prompt, encoding="utf-8")
    (LOGS_DIR / f"{run_id}-plan.json").write_text(json.dumps(plan_dict, indent=2), encoding="utf-8")

    return {
        "prediction": pred,
        "shap_top": shap_feats,
        "lime_top": lime_feats,
        "plan": plan_dict,
        "llm_source": source,  # "gemini" or "fallback"
        "run_id": run_id,
    }

def generate_intervention_for_index(i: int, X: pd.DataFrame, model, label_map: Dict[int,str],
                                    doc_index: Optional[DocIndex], tone: Tone="supportive",
                                    k_policy: int=5, k_features: int=3) -> Dict[str,Any]:
    row_df = X.iloc[[i]]; row = X.iloc[i]
    pred = predict_one(model, row_df, label_map)
    bg   = X.sample(min(100, len(X)), random_state=42) if len(X) > 1 else X
    shap_feats = shap_top_k(model, row_df, bg, k=k_features)
    lime_feats = lime_top_k(model, X, row_df, k=k_features)
    policy_ctx = retrieve_policy_ctx(doc_index, pred["pred_label"], shap_feats, lime_feats, pred["proba_dict"], k=k_policy) if doc_index else ""
    citations  = extract_citations(policy_ctx)
    prompt     = build_prompt(row, pred["pred_label"], pred["proba_dict"], shap_feats, lime_feats, policy_ctx, tone)
    client     = GeminiClient()
    plan, source = client.generate_plan(prompt)
    plan.risk_level = _risk_from_prob(float(pred["proba_dict"].get("At-Risk", 0.0)))
    if not plan.citations and citations:
        plan.citations = citations
    plan_dict = plan.model_dump(mode="json")

    run_id = dt.datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + str(uuid.uuid4())[:8]
    (LOGS_DIR / f"{run_id}-prompt.txt").write_text(prompt, encoding="utf-8")
    (LOGS_DIR / f"{run_id}-plan.json").write_text(json.dumps(plan_dict, indent=2), encoding="utf-8")

    return {
        "index": i,
        "prediction": pred,
        "shap_top": shap_feats,
        "lime_top": lime_feats,
        "plan": plan_dict,
        "llm_source": source,
        "run_id": run_id,
    }
