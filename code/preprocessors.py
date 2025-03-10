#######################################
#  Data Preprocessing & Adjacency     #
#######################################
import json
import os
import uuid

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

#######################################
#         Global Variables            #
#######################################
# Unique Run ID
run_id = str(uuid.uuid4())

script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_dir, "config.json")

# Open the file using the absolute path
with open(config_path, "r") as config_file:
    config = json.load(config_file)

THRESHOLD = config['threshold']
NOISE_FACTOR = config['noise_factor']


######################################
#  Normalising Gene Expression Data  #
######################################
def preprocess_data(file_path):
    # Determine separator based on file extension
    sep = "\t" if file_path.endswith(".tsv") else ","
    # Load the file and use the first column as the index (gene names)
    df = pd.read_csv(file_path, sep=sep, index_col=0)
    gene_names = df.index.values
    expr_values = df.values
    scaler = StandardScaler()
    normalized_expr = scaler.fit_transform(expr_values.T).T
    return normalized_expr, gene_names


#####################################################
#  Creating adjacency matrix as prior for training  #
#####################################################
def construct_adjacency_matrix(expr_data, threshold):
    corr_matrix = np.corrcoef(expr_data)
    adj_matrix = (corr_matrix > threshold).astype(float)
    np.fill_diagonal(adj_matrix, 0)
    return adj_matrix


def construct_adjacency_matrix_with_noise(expr_data, threshold, noise_factor=NOISE_FACTOR):
    adj_matrix = construct_adjacency_matrix(expr_data, threshold)
    random_noise = np.random.rand(*adj_matrix.shape) < noise_factor
    return np.logical_or(adj_matrix, random_noise).astype(float)


####################################################################
#  Creating adjacency matrix from the ground truth (if available)  #
####################################################################
def create_true_adjacency_matrix(network_file, gene_names):
    # Each row in the network file represents one unique edge.
    net_df = pd.read_csv(network_file)
    true_adj_matrix = pd.DataFrame(0, index=gene_names, columns=gene_names)
    for _, row in net_df.iterrows():
        gene1, gene2 = row['Gene1'], row['Gene2']
        true_adj_matrix.at[gene1, gene2] = 1
        true_adj_matrix.at[gene2, gene1] = 1
    return true_adj_matrix
