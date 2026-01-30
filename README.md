# Synthetic_Dataset

This synthetic dataset is designed to approximate the structure and statistical properties of the HAALSI dataset. Its primary purpose is to provide a controlled setting for understanding and interpreting the results of the bootstrapped clustering pipeline, as well as for evaluating the impact of different preprocessing choices on clustering outcomes by providing a previously known “ground truth”.

# Installation
```
pip install numpy pandas scikit-learn
git clone https://github.com/GlobalHealthML-UNIHID/Synthetic_Dataset.git
cd Synthetic_Dataset
```


# Data Generation Technique

Synthetic data are generated using a region-conditioned latent factor model. Each individual is first assigned to one of three regions (rural, mixed, urban), representing a shared environmental context. Given this assignment, two continuous latent factors are used to induce structured correlations across subsets of biomarkers. The biomarkers are generated as noisy functions of these latent factors, with region and sex specific offsets applied to selected biomarkers. Biomarkers are generated as:

Value = Baseline Value + Latent effect + Region shift + Sex shift  + Noise

with some terms set to zero where applicable. 

ex:
<img width="1001" height="159" alt="image" src="https://github.com/user-attachments/assets/d4dd5090-00e1-4917-b1a6-f6390d704077" />


# Repository Structure

**1. generator.py:**  Implements the statistical data-generating process. Applies latent factor loadings, region/sex shifts, and noise. Accepts a scene configuration as input.
Returns biomarker matrix X, metadata, and ground-truth variables. Contains no experiment-specific parameterization.

**2. scene files:** Defines a fully reproducible synthetic population.Specifies sample size, seed, region/sex distributions, and latent structure. Configures biomarker loadings, shifts, and noise parameters. Optionally defines explicit ground-truth cluster structure. Provides the complete configuration for generator.py.
<img width="679" height="718" alt="image" src="https://github.com/user-attachments/assets/0bf5b63a-3760-477f-aa53-dd46b60cc86d" />

**3. Run Scene:** executes a complete synthetic experiment from a scene configuration. It loads a scene module, generates the dataset, saves outputs, and optionally runs the bootstrapped clustering pipeline. 

# Demo files

An example scene configuration has been provided so that the different preprocessing choices can be visualised via the clustering outputs.

```
cd Synthetic_data_set
python -m scenes.run_scene --scene scenes.scene_example
```
To Generate data + cluster:
``` python -m scenes.run_scene --scene scenes.scene_example```
   
To generate data only:
``` python -m scenes.run_scene --scene scenes.scene_example --no-cluster
```


