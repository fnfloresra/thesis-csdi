# CSDI - Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation

This repository contains an implementation of CSDI for the thesis project on data imputation using diffusion models.

Based on the NeurIPS 2021 paper "[CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation](https://arxiv.org/abs/2107.03502)".

## Requirements

Please install the packages in requirements.txt:

```shell
pip install -r requirements.txt
```

## Preparation

### Download the healthcare dataset 
```shell
python download.py physio
```

### Download the air quality dataset 
```shell
python download.py pm25
```

## Experiments 

### Training and imputation for the healthcare dataset
```shell
python exe_physio.py --testmissingratio [missing ratio] --nsample [number of samples]
```

### Imputation for the healthcare dataset with pretrained model
```shell
python exe_physio.py --modelfolder pretrained --testmissingratio [missing ratio] --nsample [number of samples]
```

## Project Structure

```
.
├── config/                 # Configuration files
│   └── base.yaml          # Default configuration
├── data/                  # Data directory (created after download)
├── save/                  # Model checkpoints directory
├── diff_models.py         # Diffusion model components
├── main_model.py          # Core CSDI model implementation
├── dataset_physio.py      # Healthcare dataset loader
├── download.py            # Data download utility
├── exe_physio.py          # Execution script for healthcare data
├── utils.py               # Training and evaluation utilities
├── requirements.txt       # Python dependencies
├── LICENSE                # MIT License
└── README.md              # This file
```

## Citation

If you use this code for your research, please cite the original paper:

```bibtex
@inproceedings{tashiro2021csdi,
  title={CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation},
  author={Tashiro, Yusuke and Song, Jiaming and Song, Yang and Ermon, Stefano},
  booktitle={Advances in Neural Information Processing Systems},
  year={2021}
}
```

## Acknowledgements

This implementation is based on the original [CSDI repository](https://github.com/ermongroup/CSDI) by ermongroup.

A part of the codes is based on [BRITS](https://github.com/caow13/BRITS) and [DiffWave](https://github.com/lmnt-com/diffwave)