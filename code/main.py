# ---------------------------
# Objective Function for Exhaustive Search
# ---------------------------
import json
import os
import time
import uuid

import numpy as np
import optuna
import torch
from optuna.samplers import GridSampler
from sklearn.metrics import accuracy_score

from visualizers import analyze_network_properties
from evaluators import train_and_evaluate, calculate_early_precision_rate, calculate_whole_network_overlap, evaluate_model

from loggers import log_run_info
from preprocessors import preprocess_data, construct_adjacency_matrix_with_noise, create_true_adjacency_matrix, construct_adjacency_matrix
from model import GAT_VGAE
import torch.nn.functional as F

#######################################
#         Global Variables            #
#######################################
# Unique Run ID
run_id = str(uuid.uuid4())

# Load configuration from file
with open('config.json', 'r') as config_file:
    config = json.load(config_file)

# File paths and hyperparameters from config
RUN_INFO_PATH = config['run_info_path']
EPOCH_INFO_PATH = config['epoch_info_path']
EXPR_FILE = config['expr_file']
NETWORK_FILE = config['network_file']
DATASET = config['dataset']

NUM_NEURONS = config['num_neurons']
EMBEDDING_SIZE = config['embedding_size']
NUM_HEADS = config['num_heads']
LEARNING_RATE = config['learning_rate']
NUM_EPOCHS = config['num_epochs']
THRESHOLD = config['threshold']
NOISE_FACTOR = config['noise_factor']

# Additional configurable values
DROPOUT = config.get('dropout', 0.2)
K_FRACTION = config.get('k_fraction', 0.1)  # Used in EPR calculation

TUNE_HYPERPARAMETERS = config['tune_hyperparameters']
GROUND_TRUTH_AVAILABLE = config.get("ground_truth_available", True)


def objective(trial):
    num_neurons = trial.suggest_categorical('num_neurons', [16, 32, 64, 128])
    embedding_size = trial.suggest_categorical('embedding_size', [8, 16, 32, 64])
    num_heads = trial.suggest_categorical('num_heads', [2, 4, 8, 16])
    learning_rate = trial.suggest_categorical('learning_rate', [0.001, 0.0001, 0.005, 0.0005])

    print(f"Trial {trial.number}: num_neurons={num_neurons}, embedding_size={embedding_size}, "
          f"num_heads={num_heads}, learning_rate={learning_rate:.5f}")

    expr_data, gene_names = preprocess_data(EXPR_FILE)
    adj_matrix = construct_adjacency_matrix_with_noise(expr_data, threshold=THRESHOLD, noise_factor=NOISE_FACTOR)
    expr_tensor = torch.FloatTensor(expr_data)
    edge_index = torch.tensor(np.array(np.where(adj_matrix == 1)), dtype=torch.long)
    adj_matrix_tensor = torch.FloatTensor(adj_matrix)
    if GROUND_TRUTH_AVAILABLE:
        true_adj_matrix = create_true_adjacency_matrix(NETWORK_FILE, gene_names)
    else:
        true_adj_matrix = None
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    edge_index = edge_index.to(device)
    expr_tensor = expr_tensor.to(device)
    adj_matrix_tensor = adj_matrix_tensor.to(device)

    model = GAT_VGAE(
        num_features=expr_data.shape[1],
        num_neurons=num_neurons,
        embedding_size=embedding_size,
        num_heads=num_heads,
        num_nodes=adj_matrix.shape[0]
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_and_evaluate(model, edge_index, expr_tensor, adj_matrix_tensor, true_adj_matrix, num_epochs=NUM_EPOCHS,
                       optimizer=optimizer, num_neurons=num_neurons, embedding_size=embedding_size, lr=learning_rate,
                       heads=num_heads)

    reconstructed_adjacency_eval = model(edge_index, expr_tensor).detach().cpu().numpy()
    if GROUND_TRUTH_AVAILABLE:
        roc_auc_eval, prec, rec, f1, epr, acc, num_gt_edges, n, overlap_count = evaluate_model(true_adj_matrix,
                                                                                               reconstructed_adjacency_eval)

        print(f"Trial {trial.number} completed with ROC-AUC: {roc_auc_eval:.4f}")
        log_run_info(run_id, num_neurons, embedding_size, learning_rate, num_heads,
                     roc_auc_eval, prec, rec, f1, epr, acc, num_gt_edges, n, overlap_count, DATASET)
        return epr
    else:
        # Use negative reconstruction loss as the objective (lower loss is better)
        reconstructed_tensor = torch.FloatTensor(reconstructed_adjacency_eval).view(-1)
        reconstruction_loss = F.binary_cross_entropy(reconstructed_tensor, adj_matrix_tensor.view(-1)).item()
        return -reconstruction_loss



#######################################
#          Main Execution Block       #
#######################################
if __name__ == "__main__":
    if __name__ == "__main__":
        if TUNE_HYPERPARAMETERS:
            search_space = {
                "num_neurons": [16, 32, 64, 128],
                "embedding_size": [8, 16, 32, 64],
                "num_heads": [2, 4, 8, 16],
                "learning_rate": [0.001, 0.0005]
            }
            sampler = GridSampler(search_space)
            study = optuna.create_study(direction="maximize", sampler=sampler)
            study.optimize(objective, n_trials=256)

            best_params = study.best_trial.params
            print(f"Using best hyperparameters from tuning: {best_params}")

        else:
            best_params = {
                "num_neurons": config["num_neurons"],
                "embedding_size": config["embedding_size"],
                "num_heads": config["num_heads"],
                "learning_rate": config["learning_rate"]
            }
            print(f"Using predefined hyperparameters from config.json: {best_params}")

        # --- Data Preprocessing and Adjacency Construction ---
        expr_data, gene_names = preprocess_data(EXPR_FILE)
        adj_matrix = construct_adjacency_matrix_with_noise(expr_data, threshold=THRESHOLD, noise_factor=NOISE_FACTOR)
        expr_tensor = torch.FloatTensor(expr_data)
        edge_index = torch.tensor(np.array(np.where(adj_matrix == 1)), dtype=torch.long)
        adj_matrix_tensor = torch.FloatTensor(adj_matrix)
        if GROUND_TRUTH_AVAILABLE:
            true_adj_matrix = create_true_adjacency_matrix(NETWORK_FILE, gene_names)
        else:
            true_adj_matrix = None
        # --- Model Setup ---
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        edge_index = edge_index.to(device)
        expr_tensor = expr_tensor.to(device)

        # <-- Use best_params instead of your global config variables:
        model = GAT_VGAE(
            num_features=expr_data.shape[1],
            num_neurons=best_params["num_neurons"],
            embedding_size=best_params["embedding_size"],
            num_heads=best_params["num_heads"],
            num_nodes=adj_matrix.shape[0],
            dropout=DROPOUT
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=best_params["learning_rate"])

        # --- Training ---
        start_time = time.time()
        train_and_evaluate(
            model, edge_index, expr_tensor, adj_matrix_tensor, true_adj_matrix,
            NUM_EPOCHS, optimizer,
            num_neurons=best_params["num_neurons"],
            embedding_size=best_params["embedding_size"],
            lr=best_params["learning_rate"],
            heads=best_params["num_heads"]
        )
        total_time = time.time() - start_time
        print(f"\nTotal training time: {total_time:.4f} s")

        # --- Final Evaluation ---
        reconstructed_adjacency = model(edge_index, expr_tensor).detach().cpu().numpy()
        if GROUND_TRUTH_AVAILABLE:
            roc_auc, prec, rec, f1, epr, acc, num_gt_edges, n, overlap_count = evaluate_model(true_adj_matrix,
                                                                                              reconstructed_adjacency)

            # Compute accuracy by thresholding at 0.5
            true_flat = true_adj_matrix.values.flatten()
            pred_flat = reconstructed_adjacency.flatten()
            accuracy = accuracy_score(true_flat, (pred_flat > 0.5).astype(int))

            # Compute EPR with the existing function
            epr_final = calculate_early_precision_rate(reconstructed_adjacency, true_adj_matrix)

            print(f"\nFinal ROC-AUC: {roc_auc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}, "
                  f"Accuracy: {accuracy:.4f}, EPR: {epr_final:.4f}")
            print(f"\nNumber of ground truth unique edges (from network file): {num_gt_edges}")
            print(f"Number of overlapping edges in top 20% predictions: {overlap_count}")
        else:
            # Without ground truth, simply report reconstruction loss
            reconstructed_tensor = torch.FloatTensor(reconstructed_adjacency).view(-1)
            reconstruction_loss = F.binary_cross_entropy(reconstructed_tensor, adj_matrix_tensor.view(-1)).item()
            print(f"\nFinal Reconstruction Loss: {reconstruction_loss:.4f}")

        # --- Downstream Network Analysis ---
        # Regardless of ground truth, you can analyze the predicted network
        print("\nPerforming downstream network analysis on the predicted network:")
        predicted_network = analyze_network_properties(reconstructed_adjacency, gene_names, threshold=0.6)

os.system("afplay /System/Library/Sounds/Glass.aiff")
