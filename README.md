# ML for VLSI Hotspot Detection: Ultra-High Recall Pipeline

An optimized Machine Learning pipeline designed to detect lithography hotspots in physical VLSI layouts using the ICCAD-2012 benchmark dataset.

## The Engineering Problem
In semiconductor manufacturing, a **False Negative** (missing a hotspot) results in a defective mask layer, leading to catastrophic chip failure and millions of dollars in lost wafer costs. A **False Positive** (false alarm) simply requires a few extra seconds of localized DRC/DFM simulation. 

Therefore, standard ML accuracy is a flawed metric for this domain. This pipeline abandons the default `0.5` classification threshold, utilizing probabilistic calibration to prioritize **manufacturing safety and ultra-high recall**.

## Key Performance Metrics
By shifting the decision threshold to `0.24`, the Random Forest classifier aggressively minimizes False Negatives while maintaining feature stability:
* **AUC-ROC:** 0.941
* **Hotspot Recall:** 90.1% (Caught 529 more critical defects than the baseline)
* **False Negatives:** Slashed by nearly 50% compared to default thresholding

## System Architecture

### 1. 43-Dimensional Feature Extraction
Raw layout matrices are highly sparse. This pipeline extracts dense structural representations before passing them to the classifier:
* **Discrete Cosine Transform (DCT):** Captures high/low-frequency spatial energy gradients.
* **Topological Mapping:** Extracts component area, aspect ratios, and centroid distances using connected-component labeling.
* **Geometric Gradients:** Utilizes Sobel filtering to calculate localized edge density and Laplacian variance.

### 2. High-Recall Classifier
* Algorithm: `RandomForestClassifier`
* Addressed the severe class imbalance of the ICCAD dataset using custom class-weighting penalties (`{0: 1, 1: 5}`).
* Iterative threshold optimization loop to dynamically locate the probabilistic cutoff that guarantees >90% recall.

## How to Run
1. Download the ICCAD-2012 benchmark dataset and place it in the `iccad-official/` directory.
2. Install dependencies: `pip install -r requirements.txt`
3. Execute the pipeline: `python src/pipeline.py`
