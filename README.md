# Synthetic_Dataset

This synthetic dataset is designed to approximate the structure and statistical properties of the HAALSI dataset. Its primary purpose is to provide a controlled setting for understanding and interpreting the results of the bootstrapped clustering pipeline, as well as for evaluating the impact of different preprocessing choices on clustering outcomes by provided a previously known “ground truth”.


# Data Generation Technique

Synthetic data are generated using a region-conditioned latent factor model. Each individual is first assigned to one of three regions (rural, mixed, urban), representing shared environmental context. Given this assignment, two continuous latent factors are used to induce structured correlations across subsets of biomarkers. The biomarkers are generated as noisy functions of these latent factors with region and sex specific offsets applied to selected biomarkers. Biomarkers are generated as:

Value = Baseline Value + Latent effect + Region shift + Sex shift  + Noise

with some terms set to zero where applicable. 
