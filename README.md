# CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation

This repository implements the CSDI method for time series data imputation, based on the paper "[CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation](https://arxiv.org/abs/2107.03502)" (NeurIPS 2021).

## Overview

CSDI is a score-based diffusion model that leverages conditional information from observed values to impute missing time series data. The model provides probabilistic imputation, generating multiple plausible values for missing entries, allowing uncertainty quantification.

## Project Structure

```
thesis-csdi/
├── config/
│   └── base.yaml          # Default configuration
├── data/                  # Data directory (created after download)
├── save/                  # Model checkpoints and results
├── main_model.py          # Core CSDI model implementation
├── diff_models.py         # Diffusion model components
├── utils.py               # Training and evaluation utilities
├── dataset_physio.py      # Healthcare dataset loader
├── download.py            # Data download utility
├── exe_physio.py          # Execution script for healthcare data
├── requirements.txt       # Project dependencies
├── LICENSE                # MIT License
└── README.md              # This file
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/fnfloresra/thesis-csdi.git
cd thesis-csdi
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Data Preparation

### Download the healthcare dataset (PhysioNet Challenge 2012)
```bash
python download.py physio
```

### Download the air quality dataset (PM2.5)
```bash
python download.py pm25
```

## Usage

### Training and Imputation for Healthcare Dataset

```bash
python exe_physio.py --testmissingratio [missing ratio] --nsample [number of samples]
```

#### Arguments:
- `--config`: Configuration file name (default: base.yaml)
- `--device`: Device for training, e.g., 'cuda:0' or 'cpu' (default: cuda:0)
- `--seed`: Random seed for reproducibility (default: 1)
- `--testmissingratio`: Missing ratio for testing, 0.0-1.0 (default: 0.1)
- `--nfold`: Fold number for 5-fold cross-validation, 0-4 (default: 0)
- `--unconditional`: Use unconditional CSDI model
- `--modelfolder`: Folder containing pretrained model (skip training if provided)
- `--nsample`: Number of samples for evaluation (default: 100)

### Example Commands

Train and evaluate with 10% missing ratio:
```bash
python exe_physio.py --testmissingratio 0.1 --nsample 100
```

Run on CPU:
```bash
python exe_physio.py --device cpu --testmissingratio 0.1 --nsample 10
```

Use a pretrained model:
```bash
python exe_physio.py --modelfolder pretrained --testmissingratio 0.1 --nsample 100
```

## Configuration

The configuration file (`config/base.yaml`) contains:

- **Training parameters**: epochs, batch size, learning rate
- **Diffusion parameters**: number of layers, channels, attention heads, diffusion steps
- **Model parameters**: embedding dimensions, target strategy

## Results

After evaluation, results are saved in the `save/` folder:
- `model.pth`: Trained model weights
- `config.json`: Configuration used for training
- `generated_outputs_nsample*.pk`: Generated samples
- `result_nsample*.pk`: Evaluation metrics (RMSE, MAE, CRPS)

## Metrics

The model is evaluated using:
- **RMSE** (Root Mean Square Error)
- **MAE** (Mean Absolute Error)
- **CRPS** (Continuous Ranked Probability Score)

## Acknowledgements

This implementation is based on the original [CSDI repository](https://github.com/ermongroup/CSDI) by the authors and incorporates ideas from [BRITS](https://github.com/caow13/BRITS) and [DiffWave](https://github.com/lmnt-com/diffwave).

## Citation

If you use this code, please cite the original paper:

```bibtex
@inproceedings{tashiro2021csdi,
  title={CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation},
  author={Tashiro, Yusuke and Song, Jiaming and Song, Yang and Ermon, Stefano},
  booktitle={Advances in Neural Information Processing Systems},
  year={2021}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.