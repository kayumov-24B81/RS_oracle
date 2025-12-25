# Neural Oracle for Reed–Solomon Decoding

This repository contains experimental code and notebooks for studying the use of a neural network–based oracle
to improve the decoding performance of Reed–Solomon (RS) codes by predicting error (erasure) positions.

The work is focused on a hybrid decoding scheme in which a neural model predicts erasure locations based on
indirect features, and a classical Berlekamp–Massey decoder uses these predictions during decoding.

---

## Repository structure

experiments/  
Jupyter notebooks with experiments

graphs/  
Generated plots (FSR curves, comparisons)

models/  
Saved neural network checkpoints

tables/  
Tables with experimental results (CSV)

rs/  
Reed–Solomon utilities and channel simulation, basic scripts for training and evaluation/comparison

---

## Experiments

The experiments are organized as Jupyter notebooks:

01_baseline.ipynb  
Baseline experiments with the classical RS decoder.

02_model.ipynb  
Training of the basic neural oracle (MLP).

03_ablation.ipynb  
Ablation study of loss functions, BatchNorm, Dropout and threshold values.

04_results.ipynb  
Final evaluation and comparison of classical and hybrid decoders.

The notebooks generate plots and tables that are saved to the graphs/ and tables/ directories.

---

## How to run experiments

1. Install dependencies:

pip install -r requirements.txt

2. Launch Jupyter:

jupyter notebook

3. Run the notebooks in the following order:

01_baseline.ipynb  
02_model.ipynb  
03_ablation.ipynb  
04_results.ipynb  

---

## Requirements

Dependencies are listed in requirements.txt.

---

## Notes

- The repository is intended for experimental and research purposes.
- Numerical results may slightly vary between runs due to random initialization and data generation.
- The code is provided as part of an academic study and is not optimized for production use.
