# Overview
LDA-VGHB: identifying potential lncRNA-disease associations with singular value decomposition, variational graph auto-encoder, and heterogeneous Newton boosting machine

![Alt](Flowchart.png)
## Data
In this work，lncRNADisease is data1 and MNDR is data2.
## Environment

The current implementation uses Python 3.11 and CPU-only PyTorch. From the
`project8` directory, create or refresh the isolated environment with:

```powershell
powershell -ExecutionPolicy Bypass -File .\setup_cpu.ps1
```

Exact package versions are recorded in `requirements-cpu.txt`.
## Usage
Default is 5-fold cross validation from $CV_{l}$ to $CV_{ind}$ on lncRNADisease and MNDR databases. To run the model from the `project8` directory:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_cpu.ps1
```

For a quick end-to-end environment check:

```powershell
.\.venv\Scripts\python.exe .\model\smoke_test.py
```
Extracting linear features for diseases and lncRNAs by SVD, to run:
```
..\.venv\Scripts\python.exe svd_feature\extract_svd_features.py
```
Extracting nonlinear features for diseases and lncRNAs by VGAE, to run:
```
..\.venv\Scripts\python.exe nonlinear_feature\extract_nonlinear_features.py
```
