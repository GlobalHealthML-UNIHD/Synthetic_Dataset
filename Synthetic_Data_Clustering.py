import os
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from tqdm import tqdm

from BootstrappedClustering.src.clustering import BSClustering


data_raw = pd.read_csv("path to data.csv")
if "p_id" in data_raw.columns:
    data_raw = data_raw.set_index("p_id")
data_raw.index.name = "p_id"
print("Raw Synthetic Data Shape:", data_raw.shape)

columns_bio = [
    "BP_systolic", "BP_diastolic", "Weight", "Height", "Hip", "Waist", "Pulse",
    "Hemoglobin", "Blood_Glucose", "CESD_score", "PTSD_score", "Cognitive_score",
    "Cholesterol_total", "Cholesterol_hdl", "Cholesterol_ldl", "Cholesterol_trig",
]
columns_clust = [
    "BP_systolic", "BP_diastolic", "Weight", "Height", "Hip", "Waist",
    "Hemoglobin", "Blood_Glucose", "CESD_score", "PTSD_score", "Cognitive_score",
    "Cholesterol_total", "Cholesterol_hdl", "Cholesterol_trig",
]

scaling_methods = {
    "z-score": Path("path to scaled z-score csv"),
    "min-max": Path("path to scaled min-max csv"),
    "robust": Path("path to scaled robust csv"),
    "pathological": Path("path to scaled pathological csv"),
    "binary": Path("path to scaled binary csv"),
}

base_k_values = [2, 3, 4, 8, 16, 25, 40]
resolutions = [round(i * 0.1, 1) for i in range(1, 12)]  # 0.1 -> 1.1

base_output = Path("path to output directory")
base_output.mkdir(parents=True, exist_ok=True)


for scale, scaled_file in scaling_methods.items():
    print(f"\n=== Running clustering for SCALING: {scale} ===")

    if not scaled_file.exists():
        print(f"!! Missing scale file: {scaled_file}. Skipping.")
        continue

    data = data_raw.copy()
    scaled_bio = pd.read_csv(scaled_file)
    if "p_id" in scaled_bio.columns:
        scaled_bio = scaled_bio.set_index("p_id")

    # Replace biomarker columns if present
    if data.index.equals(scaled_bio.index):
        for col in columns_bio:
            if col in scaled_bio.columns:
                data[col] = scaled_bio[col]
    elif len(data) == len(scaled_bio):
        for col in columns_bio:
            if col in scaled_bio.columns:
                data[col] = scaled_bio[col].to_numpy()
    else:
        raise ValueError(
            f"Scaled file rows ({len(scaled_bio)}) do not match data rows ({len(data)}); "
            "ensure both files have matching indices or row order."
        )

    data = data.dropna(subset=columns_clust)

    # PCA whitening for clustering
    scaler = PCA(n_components=len(columns_clust), whiten=True)
    data[columns_clust] = scaler.fit_transform(data[columns_clust])

    method_out = base_output / scale
    (method_out / "clustering").mkdir(parents=True, exist_ok=True)

    for k in base_k_values:
        print(f"-> clustering k={k}")

        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        clustering = BSClustering(km).fit(
            data[columns_clust],
            n_bootstrap=2e0,
            weights=None,
        )

        k_out_dir = method_out / "clustering" / f"{k}_clusters"
        k_out_dir.mkdir(parents=True, exist_ok=True)

        n_clusters_detected = []

        for res in tqdm(resolutions, desc=f"{scale} k={k}"):
            res_dir = k_out_dir / f"res_{res}"
            res_dir.mkdir(parents=True, exist_ok=True)

            labels = clustering.predict(resolution=res, verbose=False)
            labels.index.name = "p_id"
            labels.to_csv(res_dir / "cluster_labels.csv")

            nc = len(np.unique(labels.iloc[:, 0]))
            n_clusters_detected.append(nc)

        plt.figure(figsize=(10, 4))
        sns.lineplot(x=resolutions, y=n_clusters_detected)
        plt.title(f"Toy Data - {scale} - Base k={k}")
        plt.xlabel("Resolution")
        plt.ylabel("Detected clusters")
        plt.savefig(k_out_dir / f"{scale}_k{k}.png", dpi=300, bbox_inches="tight")
        plt.close()

print("\nAll scalings processed.")
