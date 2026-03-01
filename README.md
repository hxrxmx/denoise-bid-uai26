# $\texttt{DenoiseBid}$ Experiments Reproduction

This repository contains the implementation of the $\texttt{DenoiseBid}$ method for the UAI conference. Follow the instructions below to reproduce the experiments.

## Setup
Requires Python 3.10.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Experiments

### 1. Synthetic Data
Run all cells in the following notebooks to perform experiments and generate plots:
* `notebooks/synthetic/denoise_bid_ctr_only.ipynb`
* `notebooks/synthetic/denoise_bid_joint.ipynb`

### 2. iPinYou
Data preparation requires **Python 2.7**.
1. Clone the preprocessing repository into `./data/`:
   ```bash
   git clone https://github.com/wnzhang/make-ipinyou-data ./data/make-ipinyou-data
   ```
2. Download and Unzip the iPinYou dataset from Kaggle and extract it into `./data/make-ipinyou-data/original-data/` manually.
3. Run the data processing (requires Python 2.7):
   ```bash
   cd ./data/make-ipinyou-data/
   make all
   ```
4. Once campaign folders are generated, run all cells in:
   * `notebooks/ipinyou/denoise_bid_joint.ipynb`
   * `notebooks/ipinyou/denoise_bid_ctr_only.ipynb`

### 3. BAT
All data downloading and processing are automated.
Open the corresponding notebooks `notebooks/bat/denoise_bid_ctr_only.ipynb` or `notebooks/bat/denoise_bid_joint.ipynb` and click **Run All**.

### 4. Criteo Attribution
Two Experiments are located in `notebooks/criteo_attribution_exp_low_data/` and `notebooks/criteo_attribution_exp_missed_features/`.

1. **Preprocessing & Training:** Run `dataset_processing.ipynb`. It downloads the data, trains the models, and saves `bidding_predictions.csv`.
2. **Analysis:** Run `denoise_bid.ipynb` to generate the final plots.
