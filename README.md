# ETBiDecSD: Ensemble Transitive Bidirectional Decoupled Self-Distillation for Time Series Classification

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.6%2B-blue)](https://www.python.org/)
[![Pytorch](https://img.shields.io/badge/PyTorch-1.3%2B-orange)](https://pytorch.org/)

This repository provides the implementation of ETBiDecSD ("Ensemble Transitive Bidirectional Decoupled Self-Distillation for Time Series Classification"), a novel approach for time series classification that combines ensemble learning with bidirectional knowledge distillation.

📄 **Paper Status**: Submitted to IEEE Transactions on Systems, Man, and Cybernetics: Systems Journal


## 📥 Dataset Preparation

### UCR Archive Datasets
1. Download datasets from [UCR Time Series Classification Archive](http://timeseriesclassification.com/dataset.php)
2. Unzip the downloaded datasets
3. Place them in the `UCRData` folder with the following structure:

## 🛠 Installation

1. Clone this repository:
```
git clone https://github.com/yourusername/ETBiDecSD.git
cd ETBiDecSD


2. Install required packages:

'''
pip install -r requirements.txt


## 📂 Implementation Repository Structure
'''
ETBiDecSD/
├── UCRData/ # Dataset directory
├── network/
│ ├── network.py # Model architecture
│ └── decouple_loss.py # Custom loss functions
├── processing/
│ ├── config.py # Configuration
│ ├── data.py # Data processing
│ └── augmentation.py # Data augmentation
├── main.py # Main implementation script (e.g., Train and Test)
├── requirements.txt # Dependencies
└── README.md # This file







{
Authors="Zhiwen Xiao, Huanlai Xing, Rong Qu, Hui Li, Li Feng, Bowen Zhao, and Qian Wan",
Title = "Ensemble Transitive Bidirectional Decoupled Self-Distillation for Time Series Classification,"
}
