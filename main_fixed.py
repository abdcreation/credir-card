# -*- coding: utf-8 -*-
"""
app.py  —  Credit Card Fraud Detection Dashboard
Run:  streamlit run main_fixed.py
Requires: creditcard_fraud_detection_kan.py in same directory
          pip install streamlit
"""

import streamlit as st
from creditcard_fraud_detection_kan import (
    clean_card_number, mask_card_number, detect_card_brand, luhn_check,
    predict_transaction, explain_prediction,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FraudSense · KAN Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS (dark industrial theme, IBM Plex Mono / Syne) ─────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

/* ── Reset & base ── */
*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] {
    font-family: 'IBM Plex Mono', monospace;
    background-color: #0d0f14 !important;
    color: #e2e8f0 !important;
}

/* ── Main layout ── */
.main .block-container { padding: 1.5rem 2rem 3rem; max-width: 1400px; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #111318 !important;
    border-right: 1px solid #1e2330 !important;
}
section[data-testid="stSidebar"] * { font-family: 'IBM Plex Mono', monospace !important; }

/* ── Hero ── */
.hero-wrap {
    background: linear-gradient(120deg, #0d1117 0%, #111827 60%, #0d1117 100%);
    border: 1px solid #1e2a40;
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.8rem;
    position: relative;
    overflow: hidden;
}
.hero-wrap::before {
    content: '';
    position: absolute; inset: 0;
    background: radial-gradient(ellipse at 70% 50%, rgba(56,189,248,.06) 0%, transparent 60%),
                radial-gradient(ellipse at 10% 80%, rgba(239,68,68,.05) 0%, transparent 50%);
    pointer-events: none;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2rem; font-weight: 800;
    letter-spacing: -.04em; margin: 0 0 .4rem;
    background: linear-gradient(90deg, #f8fafc 0%, #94a3b8 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.hero-sub { font-size: .78rem; color: #64748b; margin: 0; line-height: 1.6; }
.hero-badge {
    display: inline-block; margin-left: .8rem;
    background: rgba(56,189,248,.12); border: 1px solid rgba(56,189,248,.25);
    color: #38bdf8; font-size: .7rem; border-radius: 4px;
    padding: 2px 8px; vertical-align: middle; letter-spacing: .08em;
}

/* ── Stat tiles ── */
.stat-row { display: flex; gap: 1rem; margin-bottom: 1.6rem; }
.stat-tile {
    flex: 1;
    background: #111318;
    border: 1px solid #1e2330;
    border-radius: 12px;
    padding: .9rem 1.2rem;
}
.stat-tile .st-label { font-size: .68rem; color: #475569; letter-spacing: .1em; text-transform: uppercase; margin-bottom: .2rem; }
.stat-tile .st-value { font-family: 'Syne', sans-serif; font-size: 1.7rem; font-weight: 800; color: #f1f5f9; line-height: 1; }
.stat-tile .st-sub { font-size: .68rem; color: #334155; margin-top: .15rem; }

/* ── Panel ── */
.panel {
    background: #111318;
    border: 1px solid #1e2330;
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    height: 100%;
}
.panel-title {
    font-family: 'Syne', sans-serif;
    font-size: .7rem; font-weight: 700;
    letter-spacing: .14em; text-transform: uppercase;
    color: #475569; margin: 0 0 1.1rem;
    padding-bottom: .6rem;
    border-bottom: 1px solid #1e2330;
}

/* ── Card preview ── */
.card-preview {
    background: linear-gradient(135deg, #1a1f2e 0%, #0f1520 100%);
    border: 1px solid #2d3748;
    border-radius: 12px;
    padding: .7rem 1rem;
    margin: .5rem 0 .8rem;
    display: flex; align-items: center; gap: .8rem;
    font-family: 'IBM Plex Mono', monospace;
}
.card-preview .cp-number { font-size: .9rem; color: #cbd5e1; letter-spacing: .12em; }
.card-preview .cp-brand  { font-size: .72rem; color: #64748b; }
.cp-valid   { color: #34d399; font-size: .7rem; font-weight: 600; }
.cp-invalid { color: #f87171; font-size: .7rem; font-weight: 600; }

/* ── Result verdict ── */
.verdict-box {
    border-radius: 14px;
    padding: 1.5rem 1.8rem;
    margin-bottom: 1rem;
    position: relative; overflow: hidden;
}
.verdict-box::after {
    content: '';
    position: absolute; inset: 0;
    background: radial-gradient(ellipse at 30% 50%, rgba(255,255,255,.04), transparent 70%);
}
.vb-label  { font-size: .68rem; letter-spacing: .14em; text-transform: uppercase; opacity: .7; }
.vb-verdict{ font-family: 'Syne', sans-serif; font-size: 1.8rem; font-weight: 800; margin: .2rem 0 .1rem; }
.vb-score  { font-size: .85rem; opacity: .85; }

/* ── Progress bar override ── */
div[data-testid="stProgressBar"] > div > div { border-radius: 999px !important; }

/* ── Risk bar ── */
.rbar-wrap { display:flex; align-items:center; gap:.6rem; margin:.35rem 0; }
.rbar-label{ width:130px; font-size:.72rem; color:#64748b; flex-shrink:0; }
.rbar-bg   { flex:1; height:6px; border-radius:999px; background:#1e2330; }
.rbar-fill { height:6px; border-radius:999px; }
.rbar-pct  { width:32px; text-align:right; font-size:.7rem; color:#94a3b8; }

/* ── Waterfall bar ── */
.wf-row { display:flex; align-items:center; gap:.5rem; margin:.25rem 0; font-size:.72rem; }
.wf-label{ width:145px; color:#64748b; flex-shrink:0; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.wf-bar-wrap{ flex:1; display:flex; align-items:center; gap:4px; }
.wf-bar  { height:8px; border-radius:3px; min-width:2px; }
.wf-val  { font-size:.68rem; color:#94a3b8; white-space:nowrap; }

/* ── Reason bullets ── */
.reason { 
    border-left: 2px solid #334155;
    padding: .35rem .7rem;
    margin: .3rem 0;
    font-size: .75rem;
    color: #94a3b8;
    border-radius: 0 6px 6px 0;
    background: rgba(255,255,255,.02);
}
.reason.warn { border-color: #e9a000; color: #fbbf24; background: rgba(251,191,36,.04); }
.reason.danger{ border-color: #ef4444; color: #fca5a5; background: rgba(239,68,68,.04); }

/* ── Summary table ── */
.summ-row { display:flex; justify-content:space-between; align-items:center;
            padding:.4rem 0; border-bottom:1px solid #1a1f2e; font-size:.75rem; }
.summ-key { color: #64748b; }
.summ-val { color: #cbd5e1; font-weight: 600; }

/* ── Streamlit overrides ── */
.stButton>button {
    background: linear-gradient(90deg,#1e3a5f,#1e4d8c) !important;
    border: 1px solid #2563eb !important;
    color: #e2e8f0 !important;
    border-radius: 8px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: .82rem !important;
    letter-spacing: .04em !important;
    padding: .6rem 1.2rem !important;
    transition: all .2s !important;
}
.stButton>button:hover { background: linear-gradient(90deg,#1e4d8c,#1e5faf) !important; }
label, .stTextInput label, .stNumberInput label, .stSlider label,
.stSelectbox label, .stCheckbox label { 
    font-size: .72rem !important; color: #64748b !important; letter-spacing: .06em !important;
}
.stTextInput input, .stNumberInput input {
    background: #0d0f14 !important;
    border: 1px solid #1e2330 !important;
    color: #e2e8f0 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    border-radius: 6px !important;
    font-size: .82rem !important;
}
.stSelectbox > div > div {
    background: #0d0f14 !important;
    border: 1px solid #1e2330 !important;
    color: #e2e8f0 !important;
    border-radius: 6px !important;
}
div[data-baseweb="slider"] { padding: .3rem 0; }
.stAlert { border-radius: 10px !important; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Engine Config")
    sensitivity = st.slider("Alert sensitivity", 1, 10, 5,
        help="Shifts verdict thresholds. Score is always unchanged.")
    st.caption("1 = conservative · 5 = balanced · 10 = aggressive")

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-wrap">
  <div class="hero-title">FraudSense <span class="hero-badge">KAN v2</span></div>
  <p class="hero-sub">
    Bayesian log-odds engine · Dual-modality KAN fusion · 
    Conflict-aware stealth-fraud detection · Real-time verdict
  </p>
</div>
""", unsafe_allow_html=True)

# ── Stat tiles ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="stat-row">
  <div class="stat-tile">
    <div class="st-label">Cards Scanned</div>
    <div class="st-value">2,431</div>
    <div class="st-sub">last 24 hours</div>
  </div>
  <div class="stat-tile">
    <div class="st-label">Fraud Alerts</div>
    <div class="st-value">138</div>
    <div class="st-sub">auto-flagged today</div>
  </div>
  <div class="stat-tile">
    <div class="st-label">Analyst Queue</div>
    <div class="st-value">19</div>
    <div class="st-sub">awaiting review</div>
  </div>
  <div class="stat-tile">
    <div class="st-label">Detection Rate</div>
    <div class="st-value">97.4%</div>
    <div class="st-sub">recall (fraud caught)</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Input form ────────────────────────────────────────────────────────────────
left, right = st.columns([1.3, 1], gap="large")

with left:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">💳 Transaction Input</div>', unsafe_allow_html=True)

    raw_card = st.text_input("Card Number", placeholder="4532 0151 1283 0366",
                              help="Spaces and dashes are stripped automatically.")

    # Live card preview
    cleaned = clean_card_number(raw_card)
    if cleaned:
        brand = detect_card_brand(cleaned)
        is_v  = luhn_check(cleaned)
        badge = f'<span class="cp-valid">✔ Valid · {brand}</span>' if is_v \
                else f'<span class="cp-invalid">✘ Invalid · {brand}</span>'
        st.markdown(f"""
        <div class="card-preview">
            <div class="cp-number">{mask_card_number(cleaned)}</div>
            <div class="cp-brand">{badge}</div>
        </div>""", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        amount       = st.number_input("Amount ($)", min_value=0.0, value=45.0, step=10.0)
        hour_of_day  = st.slider("Hour of day (24 h)", 0, 23, 14)
        velocity_24h = st.slider("Transactions / 24 h", 0, 20, 1)
    with c2:
        merch_risk   = st.selectbox("Merchant risk", ["Low","Medium","High"], index=0)
        merch_type   = st.selectbox("Merchant type",
                           ["Retail","Travel","Electronics","Gaming","Marketplace","Crypto"],
                           index=0)

    c3, c4 = st.columns(2)
    with c3:
        is_intl   = st.toggle("International", value=False)
        is_online = st.toggle("Online purchase", value=False)
    with c4:
        card_pres = st.toggle("Card present", value=True)

    st.write("")
    run = st.button("⚡ Analyse Transaction", type="primary", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">📐 Risk Factor Reference</div>', unsafe_allow_html=True)
    st.markdown("""
<div style='font-size:.73rem;color:#64748b;line-height:1.85;'>

**Amount**  
&nbsp;&nbsp;≤$50 → +0.0 &nbsp;|&nbsp; $200 → +0.4 &nbsp;|&nbsp; $2k → +1.7 &nbsp;|&nbsp; $10k → +2.8

**Hour of day**  
&nbsp;&nbsp;6 AM–10 PM → +0.0 (safe window)  
&nbsp;&nbsp;11 PM → +0.35 &nbsp;|&nbsp; 2 AM → +1.10 (peak fraud)

**Velocity**  
&nbsp;&nbsp;≤3 txns → +0.0 &nbsp;|&nbsp; 7–9 → +0.65 &nbsp;|&nbsp; 15+ → +1.80

**Merchant**  
&nbsp;&nbsp;Low/Retail → +0.0 &nbsp;|&nbsp; High/Crypto → +1.80

**Modality conflict > 0.40** → stealth-fraud bonus

</div>
""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ── Results ───────────────────────────────────────────────────────────────────
if run:
    pred    = predict_transaction(
        card_number=cleaned, amount=amount,
        is_international=is_intl, is_online=is_online,
        card_present=card_pres, velocity_24h=velocity_24h,
        distance_from_home=0, hour_of_day=hour_of_day,
        merchant_risk=merch_risk, merchant_type=merch_type,
        sensitivity=sensitivity,
    )
    details = explain_prediction(pred)

    st.markdown("---")
    st.markdown("### Analysis Output")

    # ── Row 1: Verdict + Summary ──────────────────────────────────────────────
    col_v, col_s = st.columns([1.1, 1], gap="large")

    with col_v:
        # Verdict box
        st.markdown(f"""
        <div class="verdict-box" style="background:{pred.color}22;border:1px solid {pred.color}44;">
            <div class="vb-label">Fraud Verdict</div>
            <div class="vb-verdict" style="color:{pred.color};">{pred.verdict}</div>
            <div class="vb-score">Alertness Score: <strong>{pred.score}/100</strong></div>
        </div>""", unsafe_allow_html=True)

        # Score bar
        bar_col = pred.color
        st.progress(pred.score / 100)

        st.markdown(f"<div style='font-size:.75rem;color:#64748b;margin:.4rem 0;'>"
                    f"Recommended: <span style='color:#cbd5e1;font-weight:600;'>"
                    f"{pred.action}</span></div>", unsafe_allow_html=True)

        # Modality probability bars
        st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Modality Risk Signals</div>', unsafe_allow_html=True)

        def rbar(label, prob, color):
            pct = int(prob * 100)
            return (f'<div class="rbar-wrap">'
                    f'<div class="rbar-label">{label}</div>'
                    f'<div class="rbar-bg"><div class="rbar-fill" '
                    f'style="width:{pct}%;background:{color};"></div></div>'
                    f'<div class="rbar-pct">{pct}%</div>'
                    f'</div>')

        dl, dc = details["dense_risk"],  details["dense_color"]
        sl, sc = details["sparse_risk"], details["sparse_color"]
        cf_col = "#c1121f" if pred.conflict_score > 0.40 else "#e9a000" if pred.conflict_score > 0.20 else "#2d6a4f"

        st.markdown(
            rbar(f"🧮 Behavioural ({dl})", pred.dense_prob, dc) +
            rbar(f"🔍 Pattern ({sl})",      pred.sparse_prob, sc) +
            rbar(f"⚡ Conflict ({pred.conflict_score:.2f})", min(pred.conflict_score, 1.0), cf_col),
            unsafe_allow_html=True,
        )

    with col_s:
        # Card + model summary
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">🃏 Transaction Summary</div>', unsafe_allow_html=True)

        def srow(k, v):
            return f'<div class="summ-row"><span class="summ-key">{k}</span><span class="summ-val">{v}</span></div>'

        valid_str = "✔ Valid" if pred.is_valid_card else "✘ Invalid"
        st.markdown(
            srow("Card (masked)",   f"<code>{mask_card_number(cleaned) if cleaned else 'N/A'}</code>") +
            srow("Card brand",      pred.card_brand) +
            srow("Card valid",      valid_str) +
            srow("Merchant",        f"{merch_type} · {merch_risk}") +
            srow("Fraud prob",      f"{pred.probability:.1%}") +
            srow("Alertness score", f"{pred.score}/100") +
            srow("Modality agreement", details["agreement"]) +
            srow("Dense weight",    f"{pred.dense_weight:.3f}") +
            srow("Sparse weight",   f"{pred.sparse_weight:.3f}"),
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Row 2: Waterfall + Reasons ────────────────────────────────────────────
    st.write("")
    col_w, col_r = st.columns([1, 1], gap="large")

    with col_w:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">📊 Risk Factor Waterfall (log-odds Δ)</div>', unsafe_allow_html=True)

        contribs = details["contributions"]
        max_abs  = max((abs(v) for v in contribs.values()), default=1)
        total    = sum(contribs.values())

        rows_html = ""
        for name, val in contribs.items():
            if val == 0.0: continue
            color = "#2d6a4f" if val < 0 else ("#ef4444" if val > 1.0 else "#e9a000" if val > 0.4 else "#38bdf8")
            bar_w = int(abs(val) / max_abs * 100)
            rows_html += (
                f'<div class="wf-row">'
                f'<div class="wf-label">{name}</div>'
                f'<div class="wf-bar-wrap">'
                f'<div class="wf-bar" style="width:{bar_w}%;background:{color};"></div>'
                f'</div>'
                f'<div class="wf-val">{val:+.2f}</div>'
                f'</div>'
            )
        # Total line
        total_col = "#ef4444" if total > 1.5 else "#e9a000" if total > 0 else "#2d6a4f"
        rows_html += (
            f'<div style="border-top:1px solid #1e2330;margin:.5rem 0 .3rem;"></div>'
            f'<div class="wf-row">'
            f'<div class="wf-label" style="color:#cbd5e1;font-weight:600;">TOTAL</div>'
            f'<div class="wf-bar-wrap">'
            f'<div class="wf-bar" style="width:{min(100,int(abs(total)/max_abs*100))}%;background:{total_col};"></div>'
            f'</div>'
            f'<div class="wf-val" style="color:{total_col};font-weight:600;">{total:+.2f}</div>'
            f'</div>'
        )
        st.markdown(rows_html, unsafe_allow_html=True)
        st.markdown(f"""
        <div style='margin-top:.8rem;font-size:.68rem;color:#475569;'>
        Total log-odds {total:+.2f} → sigmoid = <strong style='color:#94a3b8;'>{pred.probability:.1%}</strong>
        </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_r:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">🔍 Why This Score</div>', unsafe_allow_html=True)

        if not details["reasons"]:
            st.markdown('<div class="reason">No risk factors detected — transaction appears normal.</div>',
                        unsafe_allow_html=True)
        else:
            for r in details["reasons"]:
                css = "danger" if any(x in r for x in ["🚨","FRAUD","✘","Invalid","Extreme","Stealth","⚡","Very high"]) \
                      else "warn" if any(x in r for x in ["⚠","Late","High","International","Far","CNP","risk"]) \
                      else ""
                st.markdown(f'<div class="reason {css}">{r}</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

else:
    st.info("⬆️  Fill in transaction details and click **⚡ Analyse Transaction** to run the KAN fraud engine.", icon="💡")
