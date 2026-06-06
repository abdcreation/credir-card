# -*- coding: utf-8 -*-
"""
creditcard_fraud_detection_kan.py
Bayesian Log-Odds + KAN Fourier Fraud Engine
All factors stack additively in log-odds space.
KAN Fourier layer provides non-linear final modulation.
Calibrated: safe baseline → score 3, max-risk → score 95+
"""
from __future__ import annotations
import math, re
from dataclasses import dataclass, field

# ── Card Utilities ────────────────────────────────────────────────────────────

def clean_card_number(raw):
    return re.sub(r"\D", "", (raw or "").strip())

def mask_card_number(card):
    card = clean_card_number(card)
    if len(card) < 4: return "*" * len(card)
    masked = "*" * (len(card) - 4) + card[-4:]
    return " ".join(masked[i:i+4] for i in range(0, len(masked), 4))

def luhn_check(card):
    digits = [int(c) for c in card if c.isdigit()]
    if len(digits) < 13: return False
    odd  = digits[-1::-2]
    even = digits[-2::-2]
    return (sum(odd) + sum(sum(divmod(d*2,10)) for d in even)) % 10 == 0

_BRANDS = [
    ("Visa",        re.compile(r"^4[0-9]{12}(?:[0-9]{3})?$")),
    ("Mastercard",  re.compile(r"^5[1-5][0-9]{14}$")),
    ("Amex",        re.compile(r"^3[47][0-9]{13}$")),
    ("Discover",    re.compile(r"^6(?:011|5[0-9]{2})[0-9]{12}$")),
    ("Diners Club", re.compile(r"^3(?:0[0-5]|[68][0-9])[0-9]{11}$")),
    ("JCB",         re.compile(r"^(?:2131|1800|35\d{3})\d{11}$")),
    ("UnionPay",    re.compile(r"^62[0-9]{14,17}$")),
    ("Maestro",     re.compile(r"^(?:5018|5020|5038|6304|6759|6761|6763)[0-9]{8,15}$")),
]
def detect_card_brand(card):
    for brand, pat in _BRANDS:
        if pat.match(card): return brand
    return "Unknown"

# ── Math ──────────────────────────────────────────────────────────────────────

def _sig(x):   return 1.0/(1.0+math.exp(-max(-25.,min(25.,x))))
def _logit(p): p=max(1e-7,min(1-1e-7,p)); return math.log(p/(1-p))
def _clamp(v,lo=0.,hi=1.): return max(lo,min(hi,v))

# ── Log-odds risk factors (additive deltas) ───────────────────────────────────
# Baseline prior: sigmoid(-3.5) ≈ 2.9%  → baseline score = 3/100

_BASE = -3.50

def _lo_amount(a):
    """$50 → 0.0 .. $10k+ → 2.80"""
    if a <= 50: return 0.0
    ref = math.log1p(50)
    scale = 2.80 / (math.log1p(10_000) - ref)
    return _clamp((math.log1p(a) - ref) * scale, 0.0, 2.80)

def _lo_hour(h):
    """Daytime → 0. Night peak 2 AM → 1.10"""
    if 6 <= h <= 22: return 0.0
    return {23:0.35, 0:1.05, 1:0.90, 2:1.10, 3:0.90, 4:0.65, 5:0.45}.get(h, 0.0)

def _lo_velocity(v):
    """Transactions/24 h. ≤3 → 0. 15+ → 1.80"""
    if v<=3: return 0.0
    if v<=6: return 0.20
    if v<=9: return 0.65
    if v<=14: return 1.20
    return 1.80

def _lo_distance(d):
    """Distance from home. <10 km → 0. 2000+ km → 1.20"""
    if d<10:   return 0.0
    if d<100:  return 0.15
    if d<500:  return 0.45
    if d<2000: return 0.85
    return 1.20

def _lo_flags(intl, online, present):
    delta, labels = 0.0, []
    if intl:
        delta += 0.40; labels.append("International")
    if online and not present:
        delta += 0.55; labels.append("Card-Not-Present Online")
    elif online and present:
        delta += 0.05; labels.append("Online (card present)")
    elif not present:
        delta += 0.25; labels.append("Card Absent In-Person")
    return delta, labels

def _lo_merchant(risk, mtype):
    r = {"Low":0.0,"Medium":0.30,"High":1.00}.get(risk, 0.40)
    t = {"Crypto":0.80,"Gaming":0.50,"Travel":0.20,
         "Electronics":0.20,"Marketplace":0.10,"Retail":0.0}.get(mtype, 0.10)
    return r + t

def _lo_card(valid, brand):
    return (3.50 if not valid else 0.0) + (0.50 if brand=="Unknown" else 0.0)

# ── KAN Fourier non-linear modulation ────────────────────────────────────────

def _kan_modulate(log_odds):
    """
    Apply FourierKAN modulation to the total log-odds.
    Sine-only (zero-centred) → compresses extreme values slightly,
    adds non-linearity without systematic inflation.
    Scale 0.04 keeps modulation < ±0.15.
    """
    return log_odds + sum(math.sin(n * log_odds) * 0.04 for n in range(1, 5))

# ── Dual-modality split (for display and conflict scoring) ───────────────────

def _dense_prob(amount, hour):
    """Behavioural modality: amount + hour. Baseline → 0.029"""
    return _sig(_BASE + _lo_amount(amount) + _lo_hour(hour))

def _sparse_prob(vel, dist, intl, online, present, risk, mtype):
    """Pattern modality: velocity, distance, flags, merchant. Baseline → 0.029"""
    fl, _ = _lo_flags(intl, online, present)
    return _sig(_BASE + _lo_velocity(vel) + _lo_distance(dist) + fl + _lo_merchant(risk, mtype))

# ── DataClass ─────────────────────────────────────────────────────────────────

@dataclass
class FraudPrediction:
    score: int; probability: float; verdict: str; action: str; color: str
    dense_prob: float; sparse_prob: float; conflict_score: float
    dense_weight: float; sparse_weight: float
    card_brand: str; is_valid_card: bool
    contributions: dict = field(default_factory=dict)
    flags: list        = field(default_factory=list)

# ── Main predict ──────────────────────────────────────────────────────────────

def predict_transaction(
    card_number, amount, is_international, is_online, card_present,
    velocity_24h, distance_from_home, hour_of_day,
    merchant_risk="Low", merchant_type="Retail", sensitivity=5,
):
    # Card
    card  = clean_card_number(card_number)
    brand = detect_card_brand(card) if card else "Unknown"
    valid = luhn_check(card) if card else False

    flags = []
    if card and not valid:        flags.append("Card failed Luhn validation")
    if card and brand=="Unknown": flags.append("Unrecognised card network")

    # Individual log-odds contributions
    lo_amount   = _lo_amount(amount)
    lo_hour     = _lo_hour(hour_of_day)
    lo_vel      = _lo_velocity(velocity_24h)
    lo_dist     = _lo_distance(distance_from_home)
    lo_fl, _    = _lo_flags(is_international, is_online, card_present)
    lo_merch    = _lo_merchant(merchant_risk, merchant_type)
    lo_card_pen = _lo_card(valid, brand) if card else 0.0

    # Total log-odds (all factors stacked)
    total_lo = _BASE + lo_amount + lo_hour + lo_vel + lo_dist + lo_fl + lo_merch + lo_card_pen

    # KAN Fourier non-linear modulation
    total_lo = _kan_modulate(total_lo)

    fused = _clamp(_sig(total_lo))

    # Dual-modality split (for display + conflict)
    dp = _dense_prob(amount, hour_of_day)
    sp = _sparse_prob(velocity_24h, distance_from_home,
                      is_international, is_online, card_present,
                      merchant_risk, merchant_type)
    conflict = round(abs(dp - sp), 4)

    # Modality weights (for display)
    lo_d, lo_s = _logit(dp), _logit(sp)
    ad, as_ = abs(lo_d), abs(lo_s)
    tot = (ad + as_) or 1.0
    wd, ws = round(ad/tot, 4), round(as_/tot, 4)

    # Human-readable flags
    if card and not valid:                flags.append("Invalid card raises fraud risk")
    if amount > 2000:                     flags.append(f"High amount: ${amount:,.0f}")
    if amount > 5000:                     flags.append(f"Very high amount: ${amount:,.0f}")
    if 0 <= hour_of_day <= 5:            flags.append(f"Late-night hour ({hour_of_day:02d}:00)")
    if velocity_24h >= 7:                 flags.append(f"High velocity: {velocity_24h} txns/24 h")
    if velocity_24h >= 12:               flags.append("Extreme velocity — possible account takeover")
    if distance_from_home >= 500:        flags.append(f"Far from home: {distance_from_home:,} km")
    if is_international:                 flags.append("International transaction")
    if is_online and not card_present:   flags.append("Card-not-present online (peak CNP risk)")
    if merchant_type in {"Crypto","Gaming"}: flags.append(f"High-risk merchant: {merchant_type}")
    if merchant_risk == "High":          flags.append("Merchant rated High-risk")
    if conflict > 0.40:                  flags.append(f"⚡ Stealth-fraud signal — modality conflict {conflict:.2f}")

    contributions = {
        "Baseline prior":   _BASE,
        "Amount":           lo_amount,
        "Hour of day":      lo_hour,
        "Velocity":         lo_vel,
        "Distance":         lo_dist,
        "Flags (intl/CNP)": lo_fl,
        "Merchant":         lo_merch,
        "Card validity":    lo_card_pen,
    }

    score = round(fused * 100)

    # Hard rule: Invalid card (Luhn failure) is immediately flagged as HIGH RISK
    if card and not valid:
        fused = max(fused, 0.75)  # Force fused probability ≥ 75% (score ≥ 75/100)
        score = round(fused * 100)

    # Sensitivity shifts thresholds only — never the raw score
    delta = (sensitivity - 5) * 0.03
    ft = _clamp(0.65 - delta, 0.45, 0.85)  # fraud threshold
    st = _clamp(0.28 - delta, 0.12, 0.45)  # suspicious threshold

    if fused >= ft:
        verdict, action, color = "🚨 FRAUD DETECTED", "Block immediately and notify cardholder.", "#c1121f"
    elif fused >= st:
        verdict, action, color = "⚠️ SUSPICIOUS", "Hold for review; request step-up auth.", "#e9a000"
    else:
        verdict, action, color = "✅ LEGITIMATE", "Approve — no action required.", "#2d6a4f"

    return FraudPrediction(
        score=score, probability=round(fused,4), verdict=verdict,
        action=action, color=color,
        dense_prob=round(dp,4), sparse_prob=round(sp,4),
        conflict_score=conflict, dense_weight=wd, sparse_weight=ws,
        card_brand=brand, is_valid_card=valid,
        contributions=contributions, flags=flags,
    )

# ── Explanation ───────────────────────────────────────────────────────────────

_BANDS = [
    (0.00, 0.15, "Very Low",  "#2d6a4f"),
    (0.15, 0.30, "Low",       "#52b788"),
    (0.30, 0.50, "Moderate",  "#e9a000"),
    (0.50, 0.70, "High",      "#e05c1a"),
    (0.70, 1.01, "Very High", "#c1121f"),
]

def _rlabel(p):
    for lo,hi,lbl,col in _BANDS:
        if lo<=p<hi: return lbl,col
    return "Very High","#c1121f"

def explain_prediction(pred):
    dl, dc = _rlabel(pred.dense_prob)
    sl, sc = _rlabel(pred.sparse_prob)
    c = pred.conflict_score
    agreement = "High" if c<0.15 else "Moderate" if c<0.35 else "Low — potential stealth fraud"
    reasons = list(pred.flags)
    if pred.dense_prob > pred.sparse_prob + 0.10:
        reasons.append(f"Behavioural signals ({pred.dense_prob:.0%}) more alarming than pattern ({pred.sparse_prob:.0%}).")
    elif pred.sparse_prob > pred.dense_prob + 0.10:
        reasons.append(f"Pattern signals ({pred.sparse_prob:.0%}) dominate over behavioural ({pred.dense_prob:.0%}).")
    else:
        reasons.append(f"Both modalities agree — behavioural {pred.dense_prob:.0%}, pattern {pred.sparse_prob:.0%}.")
    if not reasons:
        reasons.append("No specific risk factors — transaction appears normal.")
    return dict(
        dense_risk=dl, dense_color=dc, sparse_risk=sl, sparse_color=sc,
        agreement=agreement, reasons=reasons, contributions=pred.contributions,
    )

# ── Self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    TESTS = [
        dict(label="Safe baseline — valid Visa, $45, daytime, domestic",
             kw=dict(card_number="4532015112830366",amount=45,is_international=False,
                     is_online=False,card_present=True,velocity_24h=1,distance_from_home=2,
                     hour_of_day=14,merchant_risk="Low",merchant_type="Retail",sensitivity=5),
             expect="✅", max_score=10),
        dict(label="Normal online — $180, evening, Electronics",
             kw=dict(card_number="5500005555555559",amount=180,is_international=False,
                     is_online=True,card_present=True,velocity_24h=2,distance_from_home=10,
                     hour_of_day=20,merchant_risk="Low",merchant_type="Electronics",sensitivity=5),
             expect="✅", max_score=15),
        dict(label="Borderline — intl, CNP, Travel, 23h",
             kw=dict(card_number="4532015112830366",amount=700,is_international=True,
                     is_online=True,card_present=False,velocity_24h=5,distance_from_home=350,
                     hour_of_day=23,merchant_risk="Medium",merchant_type="Travel",sensitivity=5),
             expect="⚠️", min_score=30),
        dict(label="Max-risk — all danger signals",
             kw=dict(card_number="4532015112830366",amount=4999,is_international=True,
                     is_online=True,card_present=False,velocity_24h=15,distance_from_home=3200,
                     hour_of_day=2,merchant_risk="High",merchant_type="Crypto",sensitivity=5),
             expect="🚨", min_score=95),
    ]
    all_ok = True
    for t in TESTS:
        p = predict_transaction(**t["kw"])
        ok = True
        if t.get("expect") and not p.verdict.startswith(t["expect"]): ok=False
        if t.get("max_score") is not None and p.score > t["max_score"]: ok=False
        if t.get("min_score") is not None and p.score < t["min_score"]: ok=False
        if not ok: all_ok=False
        print(f"{'✅ PASS' if ok else '❌ FAIL'}  {t['label']}")
        print(f"       verdict={p.verdict}  score={p.score}/100  dense={p.dense_prob:.3f}  sparse={p.sparse_prob:.3f}  conflict={p.conflict_score:.3f}\n")
    print("✅ All passed." if all_ok else "❌ Some tests failed.")