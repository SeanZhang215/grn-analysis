# GRN Analysis

A pipeline for constructing and analyzing Gene Regulatory Networks (GRNs) using bulk RNA-seq data, with gene knockout effect prediction.

## Overview

This repository contains a comprehensive pipeline for:
1. Processing and analyzing bulk RNA-seq data
2. Constructing GRNs using GRNBoost2
3. Predicting gene knockout effects using GenKI
4. Visualizing and analyzing network properties

## Repository Structure

```
grn-analysis/
├── README.md
├── requirements.txt
├── src/
│   ├── data_preparation.py      # Data preprocessing and filtering
│   ├── grn_construction.py      # GRNBoost2 network construction 
│   ├── genki_inference.py       # GenKI knockout analysis
│   └── visualization.py         # Network visualization utilities
└── notebooks/
    ├── 1_data_exploration.ipynb     # Data loading, QC, and preprocessing
    ├── 2_grn_analysis.ipynb         # Network construction and evaluation
    └── 3_knockout_effects.ipynb     # GenKI analysis and results
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/SeanZhang215/grn-analysis.git
cd grn-analysis
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate 
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Follow the notebooks in order:

1. `1_data_exploration.ipynb`: Preprocess and explore expression data
2. `2_grn_analysis.ipynb`: Construct and analyze GRNs
3. `3_knockout_effects.ipynb`: Predict and validate knockout effects

## Key Features

- Pipeline for GRN analysis
- Multi-tissue comparison support
- Comprehensive visualization tools
- Validation using GenKI framework
- Configurable parameters for analysis

## Credits

This project uses the following key algorithms and frameworks:

- **GRNBoost2**: A tree-based algorithm for GRN inference. Part of the SCENIC pipeline developed by Aibar et al. For more information, see:
  - [arboreto](https://github.com/aertslab/arboreto)
  - Paper: [SCENIC: Single-cell regulatory network inference and clustering](https://www.nature.com/articles/nmeth.4463)

- **GenKI**: A deep learning framework for gene knockout inference developed by Yang et al. For more information, see:
  - [GenKI GitHub](https://github.com/yjgeno/GenKI)
  - Paper: [GenKI: A method for imputing gene knockout effects leveraging both healthy and disease samples](https://doi.org/10.1093/bioinformatics/btac522)

## Dependencies

Major dependencies include:
- Python >= 3.8
- scanpy
- anndata
- torch
- networkx
- arboreto
- GenKI

For a complete list, see `requirements.txt`.

## License

This project is licensed under the MIT License - see the LICENSE file for details.