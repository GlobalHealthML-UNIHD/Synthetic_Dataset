
SCENE = {

    # Global
    
    "name": "haalsi_scene_v1",
    "seed": 42,
    "N": 10_000,

    # Group definitions
    "regions": ["rural", "mixed", "urban"],
    "region_probs": [0.3, 0.4, 0.3],

    "sexes": ["female", "male"],
    "sex_probs": [0.5, 0.5],

    # Latent factors
 
    # Means depend on region, creates cluster structure
    "X_mu": {
        "rural": -1.3,
        "mixed":  0.0,
        "urban":  1.3,
    },
    "Y_mu": {
        "rural":  0.6,
        "mixed":  0.0,
        "urban": -0.6,
    },
    "X_sd": 0.4,
    "Y_sd": 0.4,


    # Optional: override any coefficients in generator.py
    # (If omitted, generator defaults are used)

    "params": {
        # Height 
        "height_base": 160.0,
        "height_male_shift": 12.0,
        "height_noise": 6.5,

        # Weight 
        "weight_base": 52.0,
        "weight_X_coef": 12.0,
        "weight_height_coef": 0.45,
        "weight_male_shift": 7.0,
        "weight_urban_shift": 4.0,
        "weight_rural_shift": -3.0,
        "weight_noise": 2.8,

        # Blood pressure 
        "bps_X_coef": 8.0,
        "bps_urban_shift": 16.0,
        "bps_rural_shift": -12.0,
        "bps_male_shift": 5.0,

        "bpd_X_coef": 6.0,
        "bpd_urban_shift": 12.0,
        "bpd_rural_shift": -10.0,
        "bpd_male_shift": 4.0,

        # Mental health
        "cesd_Y_coef": 10.0,
        "cesd_rural_shift": 5.0,
        "ptsd_Y_coef": 11.0,
        "ptsd_rural_shift": 4.0,

        # Glucose & cholesterol 
        "glu_X_coef": 0.35,
        "chol_total_X_coef": 0.25,
        "chol_hdl_female_shift": 0.15,
    }
}
