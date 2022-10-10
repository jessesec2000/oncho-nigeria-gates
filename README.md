# University of South Florida S. Damnosum Field Data Cleaning Pipeline

This repository contains Python code to clean various formats of field data related to capture sites of S. Damnosum.

# Prerequisites

# Install mamba

https://mamba.readthedocs.io/en/latest/installation.html

# Create Python Environment

```
cd usf-oncho-gates
mamba env create -f environment.yml
```

# Activate Python Environment

```
conda activate usf-env
```

# Pipeline Help

```
python clean_data.py --help
```

# Example Usage

```
conda activate usf-env
python clean_data.py --in_dir data/
```
