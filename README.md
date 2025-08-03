# IBM Transactions for Anti-Money Laundering

We aim to develop a **predictive model** that identifies potential **money laundering** cases by analyzing historical account transaction patterns. Anti-money laundering efforts depend on early detection of suspicious activity. This project offers a **replicable machine learning framework** using publicly available data that demonstrates:
- Transparent feature engineering (using WoE),
- High interpretability of logistic regression,
- Practical fraud detection workflow.


This project demonstrates how to:
- Preprocess transactional data,
- Engineer meaningful features,
- Apply Weight of Evidence (WoE) binning,
- Train machine learning models (e.g., logistic regression),
- Evaluate model performance using metrics like precision, recall, and AUROC.

---

## Project Structure

```
aml-fraud-detection-main/
│
├── 00_inputs/                 # Raw dataset from Kaggle
│   └── data.csv
│
├── 01_scripts/               # Jupyter Notebooks and Python scripts
│   ├── 01_data_processing.ipynb    # Cleans and prepares the dataset
│   ├── 02_modeling_woe.ipynb       # WoE binning and model training
│   ├── bmf.py                       # Custom binning model functions
│   └── utils.py                     # Helper functions
│
├── 02_outputs/              # Model outputs
│   └── predictive_columns.csv       # Final selected features
│
└── README.md
```

---

## How to Get the Data

Data Source: [Kaggle - IBM Transactions for AML](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml)

1. Create a Kaggle account and accept the competition rules (if required).
2. Download the dataset manually or via API and place `data.csv` into the `00_inputs/` directory.

---

## How to Run the Project

1. **Install Dependencies**  
   Create a virtual environment and install required packages:
   ```
   pip install -r requirements.txt
   ```

2. **Run Notebooks in Order**:
   - `01_scripts/01_data_processing.ipynb` – Cleans and formats the input data
   - `01_scripts/02_modeling_woe.ipynb` – Performs WoE binning and trains models

3. **Outputs**:
   - Final selected features and model performance saved in `02_outputs/`.
