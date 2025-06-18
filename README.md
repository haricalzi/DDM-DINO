<p align="center">
  <img src="DDM-DINO.jpeg" width="300" height="300" align="center"/>
</p>

# From ScanDDM to ART: DDM-DINO

Visual attention models have demonstrated a growing capability in predicting scanpaths, which are sequences of fixations and eye movements. Specifically, [ScanDDM](https://github.com/phuselab/scanDDM) introduced a DDM-based approach for predicting goal-directed scanpaths in a zero-shot modality, while [ART](https://github.com/cvlab-stonybrook/ART) focused on the incremental prediction of attention during language-guided object referral tasks. The present work explores the combination of these two approaches, modifying ScanDDM with the integration of [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) to address the incremental object referral task. The resulting model has been named DDM-DINO.

### Setup
Install all the requirements with `pip install -r requirements.txt`

### Usage
1. In `main.py` define the `prompt` and the `image path`
2. Run `python main.py`

### Metrics
1. Uncomment the commented libraries in `requirements.txt`
2. Install the new requirements with `pip install -r requirements.txt`
3. In `calculate_all_metrics.py` define the `parameters`
4. Run `python calculate_all_metrics.py`

## Report
More detailed informations and examples of usage can be found in the attached PDF report.

---

Project for _Natural Interaction_ and _Affective Computing_ courses, AY _2024/2025_, by [_Hari Calzi_](https://github.com/haricalzi) and [_Salvatore Ferrara_](https://github.com/2piccio2).
