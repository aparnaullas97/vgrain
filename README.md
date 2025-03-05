# V-GRAIN: Variational GRaph Attentional autoencoder for Inference of Networks
![Visualization](https://github.com/aparnaullas97/vgrain/blob/d4891c6505702a2ac915cbac5283ce0fb12803bc/docs/vgrain.png)

## Overview
This repository contains the code and implementation details for: **Inferring Gene Regulatory Networks (GRNs) using Variational Graph Autoencoders (VGAE) with Graph Attention (GAT) Layers**. The goal of this research is to introduce a novel unsupervised archticture to infer GRNs from gene expression data.

## Features
- **Graph Construction**: Builds prior networks using **Pearson correlation and mutual information**.
- **Graph-Based Deep Learning**: Uses **VGAE with GAT layers** for unsupervised learning.
- **Hyperparameter Optimization**: Implements **Optuna** for tuning model parameters.
- **Performance Evaluation**: Includes **Early Precision Rate (EPR), precision, recall, and top 20% interactions** as key evaluation metrics.
- **Multiple Experiments**: Organizes runs systematically with logs, plots, and results.


## Dependencies
- Python 3.8+
- PyTorch
- NetworkX
- Scikit-learn
- Optuna
- Pandas
- NumPy
- Matplotlib

Install dependencies via:
```bash
pip install -r requirements.txt
```


## Installation
Clone the repository and install the required dependencies:

```bash
git clone https://github.com/aparnaullas97/vgrain.git
cd vgrain
```

## Usage
### 1. Data Preparation
Ensure your dataset is structured properly. Example paths:
- Expression Dataset File: `/data/ExpressionData.csv`
- Ground Truth Network File (if avaialble): `/data/Network.csv`

### 2. Configuration File (`config.json`)
Before running the model, ensure that the `config.json` file is correctly set up. The configuration file contains key parameters that control the training and evaluation process.

#### **Parameters Explanation**
- **`run_info_path`**: Path to save run logs
- **`epoch_info_path`**: Path to save per-epoch training details
- **`expr_file`**: Path to the gene expression dataset, `/data/ExpressionData.csv`
- **`network_file`**: Path to the prior network file, `/data/Network.csv`
- **`dataset`**: Dataset name used for logging and results
- **`num_neurons`**: Number of neurons in hidden layers
- **`embedding_size`**: Size of the latent embeddings
- **`num_heads`**: Number of attention heads in GAT layers
- **`learning_rate`**: Learning rate for training the model
- **`num_epochs`**: Number of training epochs
- **`threshold`**: Threshold value to filter predicted edges in the GRN
- **`noise_factor`**: Noise factor for input perturbation (set to `0.0` if no noise is added)
- **`tune_hyperparameters`**: If `true`, enables hyperparameter optimization using Optuna
- **`ground_truth_available`**: If `true`, allows evaluation against known GRN interactions (for simulated datasets with network.csv files)

Ensure all paths are correct before running the training script.

### 3. Training the Model
Run the main script to train the V-GRAIN model:
```bash
python main.py
```
- Modify `config.json` to adjust hyperparameters, dataset paths, and model settings.

### 4. Evaluating Model Performance
After running the model, results are stored in two key files:

1. **Run Information (`run_info_vgae_*.csv`)**: 
   - Stores details of the entire training run, including hyperparameters, model configuration, and final performance metrics.
   - Useful for tracking different experiments and comparing model variations.

2. **Epoch Information (`epoch_info_vgae_*.csv`)**: 
   - Records per-epoch training details such as loss, AUROC, precision, and other performance metrics.
   - Helps analyze model convergence and identify optimal training epochs.

These files are saved in the specified paths in `config.json`:
- `run_info_path`: Stores run-level details.
- `epoch_info_path`: Logs training progress across epochs.

## Repository Structure
The project is organized into multiple subfolders for better clarity:

```
├── code/                                           # Implementation of V-GRAIN model
│   ├── config.json                                 # Tune hyperparameters of the model and set file paths
│   ├── evaluators.py                               # Helper functions to evaluate the model
│   ├── loggers.py                                  # Helper functions for logging the model training
│   ├── main.py                                     # Main function to train the model
│   ├── model.py                                    # VGAE model definition
│   ├── preprocessors.py                            # Helper functions for preprocessing the data
│   ├── utils.py                                    # Helper functions
│   ├── visualizers.py                              # Helper functions to visualise the network and analysis
|
├── data/                                           # Contains input datasets
│   ├── simulated/                                  # Includes simulated datasets
│   │   │   │   ├── expr.csv                        # Expression data file for 100 genes from scMultiSim
│   │   │   │   ├── net.csv                         # Ground truth network file for 100 genes
│   │   │   │   ├── m1139_expr.csv                  # Expression data file for 100 genes from scMultiSim
│   │   │   │   ├── m1139_net.csv                   # Ground truth network file for 100 genes
│   ├── real/                                       # Includes real datasets for downstream analysis
│   │   │   │   ├── out_Macrophages_500.csv         # Expression data file for 500 genes - macrophages
│   │   │   │   ├── out_Macrophages_1000.csv        # Expression data file for 1000 genes - macrophages
│   │   │   │   ├── out_Macrophages_2500.csv        # Expression data file for 2500 genes - macrophages
│   │   │   │   ├── out_Macrophages_full.csv        # Full Expression data file -  macrophages
│
├── files/                                          # Stores model outputs and logs
│   ├── run_info_vgae_*.csv                         # Stores hyperparameter and final run details
│   ├── epoch_info_vgae_*.csv                       # Logs loss and performance per epoch
│
├── docs/                                           # Supporting documents
│   ├── vgrain.png                                  # Model architecture
│   ├── enrichment_analysis/                  
│   │   ├── STRING.zip                              # Zip file containing analysis from STRING 
│   │   ├── gProfiler.zip                           # Zip file containing analysis from g:Profiler 
│   │   ├── metascape.zip                           # Zip file containing analysis from Metascape 
│
├── supplementary codes/                            # Supporting codes for preparation and down stream analysis
│   ├── Enrichment.R                                # Codes for reading ans analysing data from STRING, g:Profiler and Metascape
│   ├── grn2gex_scMultiSim.R                        # Code for clustering networks and simulating using scMultiSim
│
├── requirements.txt                                # List of dependencies
├── README.md                                       # Project documentation
```


## Future Work
- Integrating additional prior knowledge for network refinement.
- Extending validation with external datasets.

## Contact
For questions or collaborations, feel free to reach out:
- **Author:** Aparna Ullas
- **Email:** aparnaullas97@gmail.com
- **GitHub:** [aparnaullas97](https://github.com/aparnaullas97)

---
**Note:** This repository is part of an academic research project.
