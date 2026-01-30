
from __future__ import annotations

from typing import Dict, Any, Sequence, Optional, Tuple
import numpy as np
import pandas as pd


def _require(scene: Dict[str, Any], key: str):
    if key not in scene:
        raise KeyError(f"Scene is missing required key: {key}")
    return scene[key]


def _choice(rng: np.random.Generator, categories: Sequence[str], probs: Sequence[float], size: int) -> np.ndarray:
    probs = np.asarray(probs, dtype=float)
    if probs.ndim != 1 or len(probs) != len(categories):
        raise ValueError("probs must be 1D and match categories length")
    s = probs.sum()
    if not np.isfinite(s) or s <= 0:
        raise ValueError("probs must sum to a positive finite value")
    probs = probs / s
    return rng.choice(categories, size=size, p=probs)


def generate(scene: Dict[str, Any]) -> pd.DataFrame:
    """
    Generate synthetic dataset from a scene dictionary.

    Required scene keys (minimal):
      seed: int
      N: int
      regions: list[str]
      region_probs: list[float]
      sexes: list[str]
      sex_probs: list[float]
      X_mu: dict[str, float]
      Y_mu: dict[str, float]
      X_sd: float
      Y_sd: float

    Optional scene keys:
      params: dict of coefficients / noise values (defaults provided below)
    """
    seed = int(_require(scene, "seed"))
    N = int(_require(scene, "N"))

    regions = _require(scene, "regions")
    region_probs = _require(scene, "region_probs")

    sexes = _require(scene, "sexes")
    sex_probs = _require(scene, "sex_probs")

    X_mu = _require(scene, "X_mu")
    Y_mu = _require(scene, "Y_mu")
    X_sd = float(_require(scene, "X_sd"))
    Y_sd = float(_require(scene, "Y_sd"))

    rng = np.random.default_rng(seed)

    region = _choice(rng, regions, region_probs, N)
    sex = _choice(rng, sexes, sex_probs, N)

    # Latent factors (region-conditioned means)
    X = np.array([rng.normal(X_mu[r], X_sd) for r in region], dtype=float)
    Y = np.array([rng.normal(Y_mu[r], Y_sd) for r in region], dtype=float)

    # Coefficients/noises (defaults replicate your current code, plus fill missing biomarkers)
    p = dict(scene.get("params", {}))

    # Height
    height_base = p.get("height_base", 160.0)
    height_male_shift = p.get("height_male_shift", 12.0)
    height_noise = p.get("height_noise", 6.5)

    Height = height_base + height_male_shift * (sex == "male") + rng.normal(0.0, height_noise, N)

    # Weight 
    weight_base = p.get("weight_base", 52.0)
    weight_X_coef = p.get("weight_X_coef", 12.0)
    weight_height_coef = p.get("weight_height_coef", 0.45)
    weight_male_shift = p.get("weight_male_shift", 7.0)
    weight_urban_shift = p.get("weight_urban_shift", 4.0)
    weight_rural_shift = p.get("weight_rural_shift", -3.0)
    weight_noise = p.get("weight_noise", 2.8)

    Weight = (
        weight_base
        + weight_X_coef * X
        + weight_height_coef * (Height - height_base)
        + weight_male_shift * (sex == "male")
        + weight_urban_shift * (region == "urban")
        + weight_rural_shift * (region == "rural")
        + rng.normal(0.0, weight_noise, N)
    )

    # Waist 
    waist_base = p.get("waist_base", 64.0)
    waist_X_coef = p.get("waist_X_coef", 13.0)
    waist_urban_shift = p.get("waist_urban_shift", 8.0)
    waist_rural_shift = p.get("waist_rural_shift", -6.0)
    waist_noise = p.get("waist_noise", 2.8)

    Waist = (
        waist_base
        + waist_X_coef * X
        + waist_urban_shift * (region == "urban")
        + waist_rural_shift * (region == "rural")
        + rng.normal(0.0, waist_noise, N)
    )

    # Hip 
    hip_base = p.get("hip_base", 88.0)
    hip_X_coef = p.get("hip_X_coef", 7.0)
    hip_female_shift = p.get("hip_female_shift", 4.0)
    hip_noise = p.get("hip_noise", 2.8)

    Hip = (
        hip_base
        + hip_X_coef * X
        + hip_female_shift * (sex == "female")
        + rng.normal(0.0, hip_noise, N)
    )

    # BP systolic 
    bps_base = p.get("bps_base", 106.0)
    bps_X_coef = p.get("bps_X_coef", 8.0)
    bps_urban_shift = p.get("bps_urban_shift", 16.0)
    bps_rural_shift = p.get("bps_rural_shift", -12.0)
    bps_male_shift = p.get("bps_male_shift", 5.0)
    bps_noise = p.get("bps_noise", 4.8)

    BP_systolic = (
        bps_base
        + bps_X_coef * X
        + bps_urban_shift * (region == "urban")
        + bps_rural_shift * (region == "rural")
        + bps_male_shift * (sex == "male")
        + rng.normal(0.0, bps_noise, N)
    )

    # BP diastolic
    bpd_base = p.get("bpd_base", 66.0)
    bpd_X_coef = p.get("bpd_X_coef", 6.0)
    bpd_urban_shift = p.get("bpd_urban_shift", 12.0)
    bpd_rural_shift = p.get("bpd_rural_shift", -10.0)
    bpd_male_shift = p.get("bpd_male_shift", 4.0)
    bpd_noise = p.get("bpd_noise", 3.8)

    BP_diastolic = (
        bpd_base
        + bpd_X_coef * X
        + bpd_urban_shift * (region == "urban")
        + bpd_rural_shift * (region == "rural")
        + bpd_male_shift * (sex == "male")
        + rng.normal(0.0, bpd_noise, N)
    )

    # Pulse 
    pulse_base = p.get("pulse_base", 66.0)
    pulse_Y_coef = p.get("pulse_Y_coef", 7.0)
    pulse_rural_shift = p.get("pulse_rural_shift", 6.0)
    pulse_noise = p.get("pulse_noise", 4.0)

    Pulse = (
        pulse_base
        + pulse_Y_coef * Y
        + pulse_rural_shift * (region == "rural")
        + rng.normal(0.0, pulse_noise, N)
    )

    # Hemoglobin 
    hb_base = p.get("hb_base", 13.2)
    hb_male_shift = p.get("hb_male_shift", 1.4)
    hb_rural_shift = p.get("hb_rural_shift", -1.8)
    hb_urban_shift = p.get("hb_urban_shift", 0.9)
    hb_noise = p.get("hb_noise", 0.55)

    Hemoglobin = (
        hb_base
        + hb_male_shift * (sex == "male")
        + hb_rural_shift * (region == "rural")
        + hb_urban_shift * (region == "urban")
        + rng.normal(0.0, hb_noise, N)
    )

    # Mental health scores 
    cesd_base = p.get("cesd_base", 9.0)
    cesd_Y_coef = p.get("cesd_Y_coef", 10.0)
    cesd_rural_shift = p.get("cesd_rural_shift", 5.0)
    cesd_noise = p.get("cesd_noise", 4.0)

    CESD_score = np.clip(
        cesd_base + cesd_Y_coef * Y + cesd_rural_shift * (region == "rural") + rng.normal(0.0, cesd_noise, N),
        0.0, 60.0
    )

    ptsd_base = p.get("ptsd_base", 7.0)
    ptsd_Y_coef = p.get("ptsd_Y_coef", 11.0)
    ptsd_rural_shift = p.get("ptsd_rural_shift", 4.0)
    ptsd_noise = p.get("ptsd_noise", 4.0)

    PTSD_score = np.clip(
        ptsd_base + ptsd_Y_coef * Y + ptsd_rural_shift * (region == "rural") + rng.normal(0.0, ptsd_noise, N),
        0.0, 60.0
    )

    # Cognitive score 
    cog_base = p.get("cog_base", 102.0)
    cog_rural_shift = p.get("cog_rural_shift", -6.0)
    cog_urban_shift = p.get("cog_urban_shift", -4.0)
    cog_noise = p.get("cog_noise", 9.0)

    Cognitive_score = np.clip(
        cog_base
        + cog_rural_shift * (region == "rural")
        + cog_urban_shift * (region == "urban")
        + rng.normal(0.0, cog_noise, N),
        40.0, 130.0
    )

    # Blood glucose (mmol/L-ish)
    glu_base = p.get("glu_base", 5.2)
    glu_X_coef = p.get("glu_X_coef", 0.35)
    glu_urban_shift = p.get("glu_urban_shift", 0.15)
    glu_rural_shift = p.get("glu_rural_shift", -0.10)
    glu_noise = p.get("glu_noise", 0.60)

    Blood_Glucose = np.clip(
        glu_base
        + glu_X_coef * X
        + glu_urban_shift * (region == "urban")
        + glu_rural_shift * (region == "rural")
        + rng.normal(0.0, glu_noise, N),
        2.5, 20.0
    )

    # Cholesterol
    chol_base = p.get("chol_total_base", 5.0)
    chol_X_coef = p.get("chol_total_X_coef", 0.25)
    chol_urban_shift = p.get("chol_total_urban_shift", 0.10)
    chol_rural_shift = p.get("chol_total_rural_shift", -0.10)
    chol_noise = p.get("chol_total_noise", 0.70)

    Cholesterol_total = np.clip(
        chol_base
        + chol_X_coef * X
        + chol_urban_shift * (region == "urban")
        + chol_rural_shift * (region == "rural")
        + rng.normal(0.0, chol_noise, N),
        2.0, 12.0
    )

    hdl_base = p.get("chol_hdl_base", 1.3)
    hdl_X_coef = p.get("chol_hdl_X_coef", -0.05)
    hdl_female_shift = p.get("chol_hdl_female_shift", 0.15)
    hdl_noise = p.get("chol_hdl_noise", 0.20)

    Cholesterol_hdl = np.clip(
        hdl_base
        + hdl_X_coef * X
        + hdl_female_shift * (sex == "female")
        + rng.normal(0.0, hdl_noise, N),
        0.2, 4.0
    )

    ldl_base = p.get("chol_ldl_base", 3.0)
    ldl_X_coef = p.get("chol_ldl_X_coef", 0.22)
    ldl_urban_shift = p.get("chol_ldl_urban_shift", 0.05)
    ldl_rural_shift = p.get("chol_ldl_rural_shift", -0.05)
    ldl_noise = p.get("chol_ldl_noise", 0.60)

    Cholesterol_ldl = np.clip(
        ldl_base
        + ldl_X_coef * X
        + ldl_urban_shift * (region == "urban")
        + ldl_rural_shift * (region == "rural")
        + rng.normal(0.0, ldl_noise, N),
        0.5, 10.0
    )

    trig_base = p.get("chol_trig_base", 1.4)
    trig_X_coef = p.get("chol_trig_X_coef", 0.18)
    trig_urban_shift = p.get("chol_trig_urban_shift", 0.10)
    trig_rural_shift = p.get("chol_trig_rural_shift", -0.08)
    trig_noise = p.get("chol_trig_noise", 0.60)

    Cholesterol_trig = np.clip(
        trig_base
        + trig_X_coef * X
        + trig_urban_shift * (region == "urban")
        + trig_rural_shift * (region == "rural")
        + rng.normal(0.0, trig_noise, N),
        0.2, 15.0
    )

    # Assemble dataframe
    df = pd.DataFrame(
        {
            "region": region,
            "sex": sex,
            "BP_systolic": BP_systolic,
            "BP_diastolic": BP_diastolic,
            "Weight": Weight,
            "Height": Height,
            "Hip": Hip,
            "Waist": Waist,
            "Pulse": Pulse,
            "Hemoglobin": Hemoglobin,
            "Blood_Glucose": Blood_Glucose,
            "CESD_score": CESD_score,
            "PTSD_score": PTSD_score,
            "Cognitive_score": Cognitive_score,
            "Cholesterol_total": Cholesterol_total,
            "Cholesterol_hdl": Cholesterol_hdl,
            "Cholesterol_ldl": Cholesterol_ldl,
            "Cholesterol_trig": Cholesterol_trig,
        }
    )

    return df
