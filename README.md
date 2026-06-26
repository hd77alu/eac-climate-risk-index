# East Africa Community Climate Risk Index

This project builds a climate-risk classification workflow for East African Community (EAC) countries using monthly World Bank Climate Change Knowledge Portal data. The notebook combines exploratory data analysis, preprocessing, feature engineering, and two model families: classical tree-based classifiers and deep neural networks. The main prediction target is `tx84rr`, a temperature-based excess mortality indicator that is converted into a three-class vulnerability label.

## Dataset

- **Source:** [The World Bank Climate Change Knowledge Portal (CCKP)](https://climateknowledgeportal.worldbank.org/)
- **Countries code:** BDI, COD, KEN, RWA, SOM, SSD, TZA, UGA
- **Collection:** ERA5 0.25-degree (era5-x0.25)
- **Type:** Timeseries
- **Variables:** `tas, txx, hd30, tr23, hurs, cdd, rx5day, tx84rr`
- **Aggregation:** Monthly

## Project Goal

The goal of this project is to estimate climate-related public health vulnerability from meteorological indicators and classify each monthly observation into one of three risk levels:

- `0` - Normal / Low Risk
- `1` - Moderate Risk / Elevated Alert
- `2` - Severe Risk / Extreme Health Emergency

The notebook focuses on the EAC region and evaluates how well environmental variables can predict periods of elevated health risk.

## Project Structure

```text
eac-climate-risk-index/
|-- README.md
|-- notebook_mmt_health_risk.ipynb
|-- dataset/
	|-- cdd-eac-world-bank.csv
	|-- hd30-eac-world-bank.csv
	|-- hurs-eac-world-bank.csv
	|-- rx5day-eac-world-bank.csv
	|-- tas-eac-world-bank.csv
	|-- tr23-eac-world-bank.csv
	|-- tx84rr-eac-world-bank.csv
	|-- txx-eac-world-bank.csv
```

### Key Files

- `notebook_mmt_health_risk.ipynb` contains the full analysis pipeline, from loading the datasets to model comparison and final interpretation.
- `dataset/` stores the monthly World Bank CSV files used in the analysis.

## Data Sources and Variables

The notebook uses the following climate indicators:

| Category | Variable | Description |
| --- | --- | --- |
| Temperature | `tas` | Average mean surface air temperature |
| Temperature | `txx` | Maximum of daily maximum temperature |
| Temperature | `hd30` | Number of hot days above 30 C |
| Temperature | `tr23` | Number of tropical nights above 23 C |
| Temperature | `hurs` | Relative humidity |
| Precipitation | `cdd` | Maximum number of consecutive dry days |
| Precipitation | `rx5day` | Largest 5-day cumulative precipitation |
| Target proxy | `tx84rr` | Temperature-based excess mortality risk |

## Setup Instructions

**1. Clone or Download the Repository**
```bash
git clone https://github.com/hd77alu/eac-climate-risk-index
cd eac-climate-risk-index
```
**2. Create and activate a Python virtual environment**

**3. Install notebook dependencies:**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow
```

## How to Use the Notebook

### Method 1: Using Jupyter Notebook (Local)
```bash
git clone https://github.com/hd77alu/eac-climate-risk-index
cd eac-climate-risk-index

# Launch Jupyter Notebook
jupyter notebook

# Open the notebook: notebook_mmt_health_risk.ipynb
```
### Method 2: Using Google Colab
1. Click the "Open in Colab" badge at the top of the notebook
2. Upload the files inside the `dataset` folder to your Colab files
3. Run all cells

### Method 3: Using VS Code
1. Open VS Code
2. Install the Jupyter extension (if not already installed)
3. Open the folder `eac-climate-risk-index`
4. Click on `notebook_mmt_health_risk.ipynb`

## Data Preprocessing Logic

The notebook applies the following preprocessing steps:

1. Load each climate CSV into a separate DataFrame.
2. Convert each dataset from wide monthly format to long format using a custom `melt_world_bank_df()` function.
3. Parse the `date` column as a datetime field.
4. Merge all climate indicators on `code`, `name`, and `date` to create a unified modeling table.
5. Restrict the modeling window to the years 2000 to 2024.
6. Create the target label `vulnerability_target` from `tx84rr` using risk thresholds.

## Handling Missing Values

We applied different imputation methods based on the nature of the variables:

*   **Zero-Filling for Frequencies:** For `hd30` (Hot Days), missing values were treated as 0, assuming that if no event was recorded in a sparse historical entry, the threshold was not met.
*   **Linear Interpolation for Targets:** For the target `tx84rr`, we used linear interpolation within country groups to patch small gaps, assuming a continuous trend in mortality risk between known points.
*   **Localized Temporal Filling:** For threshold metrics like `tr23` (Tropical Nights), we used forward and backward filling to maintain regional consistency.
*   **Median Imputation (Safety Catch):** Any remaining isolated gaps in precipitation or humidity were filled with the median to prevent row loss during training while avoiding the influence of extreme outliers.

For selected feature columns, remaining gaps are filled with the column median after temporal filling steps:

- `tr23`
- `hurs`
- `cdd`
- `rx5day`

In addition, `hd30` is filled with `0.0` in the final modeling table.

### Target variable (`tx84rr`)

For the target variable, the notebook uses a combination of interpolation and boundary filling after limiting the data to the 2000-2024 window:

- linear interpolation within each country series
- backward fill and forward fill for edge cases

This is applied by country code so the temporal pattern is preserved within each geography.

## Feature Engineering

The notebook creates several derived structures from the raw climate data:

### 1. Long-format transformation

Each source CSV is reshaped from wide monthly columns into a tidy long table with one row per country-month-variable observation.

### 2. Unified monthly panel

All long-format tables are merged into a single monthly panel using the shared keys `code`, `name`, and `date`.

### 3. Risk target creation

The continuous `tx84rr` variable is converted into a three-class target:

- values up to `0.05` map to class `0`
- values from `0.05` to `0.20` map to class `1`
- values above `0.20` map to class `2`

### 4. Train-test split

The notebook performs a stratified 80/20 split to preserve class balance across training and test sets.

### 5. Feature scaling for DNN experiments

The classical tree models use the raw merged features, while the DNN pipeline applies `StandardScaler` to the training set and reuses the fitted scaler for validation and test data.

## Modeling Workflow

The notebook compares two model families:

### Classical models

- Random Forest
- Extra Trees Classifier

### Deep learning models

- Baseline multiclass DNN
- Width and depth experiments
- Batch normalization and dropout regularization experiments
- AdamW optimization experiments
- Self-normalizing neural network using SELU and AlphaDropout

For evaluation, the notebook uses accuracy, precision, recall, F1-score, ROC-AUC, confusion matrices, ROC curves, and learning curves.

## Results

The Extra Tree and the Random Forest both outperformed the deep neural network (DNN) model, likely due to the tubular structure of the data, which favors tree-based models over deep learning. Between the two classical models, the random split nature of the Extra Tree, which consequently reduces variance in a small dataset, resulted in nudging the Random Forest and achieving the best performance in predicting vulnerability risk.


### Best classical model

The Extra Trees model is the strongest overall performer in the notebook. It achieves the best balance of generalization and class separation, with strong macro F1-score and ROC-AUC performance.

### Best deep learning model

Among the neural network experiments, the wider DNN architecture performs best, but it still trails the Extra Trees model on the final test comparison.

## Conclusion

The project shows that monthly climate indicators can be used to build a useful vulnerability classification system for EAC countries, but the current data and feature representation favor ensemble learning over deep learning.

The main limitations identified in the notebook are:

- Data volume deficit for advanced deep learning.
- Severe class imbalance and majority class Bias.
- ERA5 0.25-degree spatial grid offers high-resolution meteorological tracking, but it can still struggle to capture localized microclimates, specific topographic variations, etc.
- The target variable is heavily dependent on the chosen mathematical proxies.
- Feature space currently relies 100% on environmental indicators; it lacks context on regional poverty rates, population density, and local healthcare capacity.

