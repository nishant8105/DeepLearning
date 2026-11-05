# Customer Segmentation with a Self-Organizing Map

This assignment applies a Self-Organizing Map (SOM) to customer transaction
data. The notebook cleans and explores the data, scales the features, trains a
SOM, visualizes its map and clusters, profiles the resulting segments, and
discusses business insights. It also exports segment assignments, profile
statistics, and the trained SOM artifacts.

## Files

- `assignment4.ipynb` — the analysis and modeling workflow.
- `CC GENERAL.csv` — the customer dataset used as input.
- `customer_som_segments.csv` and `segment_profile_means.csv` — exported
  customer segment labels and segment summaries.
- `som_trained_model.pkl` and `som_weights.npy` — saved SOM model and weights.

## Requirements

```bash
pip install pandas numpy matplotlib seaborn scikit-learn minisom scipy jupyter
```

## Run

Run Jupyter from this folder, open `assignment4.ipynb`, and execute the cells
in order. Keep `CC GENERAL.csv` in this folder so the notebook can find its
input data. The notebook writes the output CSV and model files alongside the
notebook.

The source data does not contain a churn label; low engagement should not be
interpreted as confirmed churn.
