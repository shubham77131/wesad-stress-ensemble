# wesad-stress-ensemble
Stress detection from wearable HRV signals using a Soft-Voting Ensemble (RF, XGBoost, SVM) on the WESAD dataset. Achieved 82.97% LOSO accuracy.

# Abstract
This project implements a robust machine learning pipeline to detect psychological stress using heart rate variability (HRV) signals from the WESAD dataset. By employing a multi-model ensemble to address the challenge of inter-subject physiological variability.

# Dataset & Features
Dataset: (Wearable Stress and Affect Detection).
Signal Type: ECG-derived Heart Rate Variability.
Feature Set: 19 statistical and frequency-domain features (e.g., MeanNN, SDNN, pNN50, HF/LF ratio).

# Methodology
# Data Extraction & Signal Processing Pipeline
The core challenge of this project was transforming high-frequency raw sensor data into a format suitable for machine learning. For this the NeuroKit2 library was utilized to handle the specialized domain of electrocardiogram (ECG) analysis.

Automated Feature Extraction: a script was developed to iterate through the WESAD dataset, using NeuroKit2 to automatically identify cardiac cycles and compute Heart Rate Variability (HRV) metrics.

Domain Transformation: This process transformed raw electrical signals into a structured dataset of 19 physiological features, capturing variations in time intervals between heartbeats.

Handling Hardware Constraints: To manage the 2.5GB dataset on a modest machine, an incremental processing strategy was implemented, saving extracted features into a consolidated wesad_global_features.csv to avoid re-processing raw signals during model iteration.

# Model Development & Optimization
The development phase focused on finding a model architecture that could handle the unique "noise" of human physiological data.

Multi-Model Benchmarking: a modular testing framework was developed to evaluate five distinct mathematical approaches (ranging from tree-based ensembles to distance-based classifiers).

Hyperparameter Tuning: For the top-performing models, a grid-search strategy was utilized. This involved testing hundreds of combinations of model configurations to find the specific settings that balanced "learning" the data without "memorizing" specific subject noise.

Ensemble Integration: Recognizing that different models succeeded on different subjects, a Soft-Voting Ensemble was engineered. This "meta-model" aggregates the probability scores of our top three performers, ensuring that the final prediction is based on a consensus rather than a single point of failure.

# Evaluation Strategy: Leave-One-Subject-Out (LOSO)
The most critical part of the methodology was the validation strategy. Standard random splitting is insufficient for physiological data because it allows the model to learn a specific individual's heart pattern.

Subject-Wise Partitioning: Leave-One-Subject-Out (LOSO) cross-validation was implemented. In each of the 15 iterations, the model was trained on 14 subjects and tested on the 15th subject it had never encountered before.

Generalization Benchmark: This method provides a "real-world" accuracy metric, simulating how the system would perform if a new user started wearing the sensor today.

Subject Sensitivity Analysis: Performance for every individual subject was tracked to identify "physiological outliers" (such as Subject S10), providing deep insight into the model's reliability across diverse human profiles.

# Benchmarking
# Internal Performance Leaderboard
To validate the architecture of the stress detection system, six distinct modeling strategies were evaluated using a Leave-One-Subject-Out (LOSO) cross-validation protocol.


| Model Name | Mean Accuracy | Best Subject Score | Worst Subject Score | Stress Precision | Stress Recall | Notes |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **kNN Baseline** | 0.7955 | 1 (S16) | 0.5161 (S10) | 0.69 | 0.75 | Good baseline, but lower accuracy compared to tree-based models. |
| **SVM Baseline** | 0.8154 | 1 (S14, S16) | 0.5484 (S10) | 0.74 | 0.69 | Highest resilience on S10 outlier (54.8%). |
| **Random Forest** | 0.8164 | 1 (S16) | 0.4194 (S10) | 0.72 | 0.75 | Strong generalist, struggles on S10. |
| **Lightgbm Baseline** | 0.8271 | 1 (S16) | 0.4839 (S10) | 0.74 | 0.75 | Leaf-wise tree growth has better score but prone to overfitting. |
| **XGBoost Baseline** | 0.8184 | 1 (S16) | 0.4839 (S10) | 0.72 | 0.74 | Boosting is slightly better than bagging. |
| **XGBoost Tuned** | 0.8251 | 1 (S16) | 0.4516 (S10) | 0.73 | 0.76 | Tuning has improved some performance. |
| **FINAL ENSEMBLE** | **0.8297** | **1 (S16)** | **0.4193 (S10)** | **0.75** | **0.74** | **Best overall accuracy; combines strengths of RF, XGB, and SVM.** |


 # Analysis of benchmarking process
The benchmarking process yielded several critical insights into the behavior of physiological data across different ML frameworks:

The Ensemble Effect: While LightGBM achieved a high standalone baseline (82.71%), the Final Ensemble successfully pushed the accuracy to its peak of 82.97%. This proves that a soft-voting consensus between RF, Tuned XGB, and SVM captures subtle physiological variances that individual tree-based models miss.

The "Outlier Resilience" Trade-off: SVM Baseline demonstrated the highest resilience on the dataset's most difficult outlier, Subject S10 (54.84%). By including SVM in the final ensemble, we ensured the model maintains a more stable decision boundary than purely boosting-based approaches.

Tuning Impact: Hyperparameter optimization of XGBoost yielded a significant jump in Stress Recall (0.76), ensuring that the system is highly sensitive to identifying true stress states—a vital requirement for health-monitoring applications.

# External Validation (WESAD Benchmarking)
Compared to state-of-the-art research on the WESAD dataset:

This Model: Achieved 82.97% accuracy using a streamlined set of 19 ECG-only HRV features.

Context: This performance is highly competitive with studies that utilize multi-modal sensor data (EDA, EMG, and Temperature), proving that high-quality feature engineering and ensemble voting can compensate for a reduced sensor array.


## Repository Structure
```text
├── src/                  # Extraction and Training Scripts
│   ├── experiments/      # Individual model baselines (kNN, LGBM, etc.)
│   ├── preprocessing.py  # Signal processing via NeuroKit2
│   └── ensemble.py       # Final 82.97% model logic
├── models/               # Final.joblib model
├── results/              # CSV performance matrices and plots
├── requirements.txt      # Required libraries
└── wesad_global_features.csv  # Final extracted feature set


# How To Reproduce The Results
# Prerequisites
Python 3.8+
Hardware: This pipeline was optimized for low-resource environments (tested on Intel Celeron w/ 8GB RAM).
Dependencies: Install required libraries via: requirements.txt

# Data Preparation
To run the extraction and training scripts, the WESAD dataset must be structured as follows:
Download the dataset from - https://archive.ics.uci.edu/dataset/465/wesad+wearable+stress+and+affect+detection
Organize the folders in the root directory: Ensure to use the absolute file paths to avoid FileNotFoundError

# Execution Flow
To replicate the final results, execute the scripts in this order:

python src/preprocessing.py: Reads the .pkl files, processes ECG signals via NeuroKit2, and generates wesad_global_features.csv.

python src/baselines.py: Runs the initial comparison of RF, SVM, XGBoost, and kNN.

python src/ensemble.py: Executes the final LOSO validation loop for the Voting Ensemble and generates the performance matrix.

# Limitations & Possibilities
Subject Variability: As noted in the results, Subject S10 remains a challenge for most architectures. Future iterations could explore personalized calibration (transfer learning) to adjust the model to an individual's unique baseline.

Feature Expansion: While HRV features proved successful, incorporating Electrodermal Activity (EDA) and Skin Temperature data from the WESAD dataset could potentially increase the classification accuracy beyond 85%.

Model Weighting: Further research into weighted voting (giving more power to SVM for specific physiological patterns) may resolve inconsistencies in outlier performance.

# Analysis of Findings
Confusion Matrix Analysis: The "Safety" of the Model
The Ensemble Confusion Matrix (LOSO) reveals a model with high "Negative Predictive Value." This is crucial for stress monitoring.

Robust Baseline Detection: The model correctly identified the "Baseline" state 516 times, with only a small fraction of relaxed moments being misclassified as stress.

Stress Sensitivity: Out of 340 total stress instances in the testing folds, the model successfully captured 256.

Balanced Errors: The False Positives (92) and False Negatives (84) are nearly symmetrical. In a clinical context, this suggests the model doesn't have a specific "bias"—it isn't overly "paranoid" about stress, nor is it "blind" to it. It provides a balanced, realistic prediction.

# Feature Importance: The Physiological Drivers
The Consolidated Feature Importance (RF + XGB) chart reveals exactly which biological signals the ensemble prioritizes.

The "NN" Dominance: The top four features are all related to the NN interval (the time between heartbeats). HRV_MinNN and HRV_MeanNN are by far the most influential.

Insight: This tells us that the absolute speed and the minimum threshold of the heart rate are the primary "switches" the model uses to detect stress.

Distribution Markers: Features like HRV_Prc20NN and HRV_Prc80NN (percentiles) show that the model isn't just looking at averages; it's looking at the extremes of how the heart rate behaves during a 60-second window.

Parasympathetic Indicators: The presence of HRV_CVSD and HRV_pNN50 in the top 10 confirms that the model is successfully using indicators of the Vagus nerve (the body's "brake" system) to determine if a subject is actually stressed or just physically active.
