#######################################
#         Visualization Functions     #
#######################################
import networkx as nx
import torch
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import precision_recall_curve


def visualize_grn(predicted_adj_matrix, gene_names, threshold=0.5):
    G = nx.Graph()
    G.add_nodes_from(gene_names)
    n = len(gene_names)
    for i in range(n):
        for j in range(i + 1, n):
            if predicted_adj_matrix[i, j] > threshold:
                G.add_edge(gene_names[i], gene_names[j], weight=predicted_adj_matrix[i, j])
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_size=500, font_size=8, edge_color="gray")
    plt.title("Predicted Gene Regulatory Network (GRN)")
    plt.show()


# def visualize_embeddings(model, expr_tensor, edge_index, method="tsne"):
#     model.eval()
#     with torch.no_grad():
#         z = model.encode(edge_index, expr_tensor).cpu().numpy()
#     if method == "pca":
#         reduced = PCA(n_components=2).fit_transform(z)
#     else:
#         reduced = TSNE(n_components=2, perplexity=10, random_state=42).fit_transform(z)
#     plt.figure(figsize=(8, 6))
#     plt.scatter(reduced[:, 0], reduced[:, 1], alpha=0.7, c='blue', edgecolors='black')
#     for i, label in enumerate(gene_names):
#         plt.text(reduced[i, 0], reduced[i, 1], label, fontsize=8, alpha=0.75)
#     plt.title(f"Gene Embeddings Visualization ({method.upper()})")
#     plt.xlabel("Component 1")
#     plt.ylabel("Component 2")
#     plt.grid()
#     plt.show()


def plot_precision_recall_curve(true_adj_matrix, reconstructed_adjacency):
    """
    Plots the precision-recall curve based on the true and predicted adjacency matrices.
    """
    true_flat = true_adj_matrix.values.flatten()
    pred_flat = reconstructed_adjacency.flatten()
    precision_vals, recall_vals, _ = precision_recall_curve(true_flat, pred_flat)
    plt.figure(figsize=(8, 6))
    plt.plot(recall_vals, precision_vals, color='blue', label='Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True)
    plt.show()


def identify_hub_genes(predicted_adj_matrix, gene_names, top_k=10, threshold=0.5):
    G = nx.Graph()
    G.add_nodes_from(gene_names)
    n = len(gene_names)
    for i in range(n):
        for j in range(i + 1, n):
            if predicted_adj_matrix[i, j] > threshold:
                G.add_edge(gene_names[i], gene_names[j], weight=predicted_adj_matrix[i, j])
    degree_centrality = nx.degree_centrality(G)
    sorted_hubs = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)
    print("\nTop Hub Genes in the Predicted GRN:")
    for gene, cent in sorted_hubs[:top_k]:
        print(f"{gene}: {cent:.4f}")
    return [gene for gene, _ in sorted_hubs[:top_k]]


def visualize_grn_with_hubs(predicted_adj_matrix, gene_names, hub_genes, threshold=0.5):
    G = nx.Graph()
    G.add_nodes_from(gene_names)
    n = len(gene_names)
    for i in range(n):
        for j in range(i + 1, n):
            if predicted_adj_matrix[i, j] > threshold:
                G.add_edge(gene_names[i], gene_names[j], weight=predicted_adj_matrix[i, j])
    node_colors = ["red" if gene in hub_genes else "blue" for gene in gene_names]
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=500, font_size=8, edge_color="gray")
    plt.title("Predicted GRN with Hub Genes Highlighted")
    plt.show()


def analyze_network_properties(predicted_adj_matrix, gene_names, threshold=0.6):
    """
    Build a NetworkX graph from the predicted adjacency matrix (using a specified threshold)
    and compute global network properties, hub genes, and communities.
    """
    G = nx.Graph()
    n = predicted_adj_matrix.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            if predicted_adj_matrix[i, j] > threshold:
                G.add_edge(gene_names[i], gene_names[j], weight=predicted_adj_matrix[i, j])
    density = nx.density(G)
    avg_clustering = nx.average_clustering(G)
    degree_cent = nx.degree_centrality(G)
    sorted_hubs = sorted(degree_cent.items(), key=lambda x: x[1], reverse=True)
    top_hubs = sorted_hubs[:10]

    print("\n--- Downstream Network Analysis ---")
    print("Network Density:", density)
    print("Average Clustering Coefficient:", avg_clustering)
    print("\nTop 10 Hub Genes (by degree centrality):")
    for gene, cent in top_hubs:
        print(f"{gene}: {cent:.4f}")

    # Community detection using the Girvan-Newman algorithm
    from networkx.algorithms.community import girvan_newman
    communities_generator = girvan_newman(G)
    top_level_communities = next(communities_generator)
    communities = sorted(map(sorted, top_level_communities))
    print("\nDetected Communities (Girvan-Newman):")
    for idx, community in enumerate(communities):
        print(f"Community {idx + 1}: {community}")

    # Degree centrality
    degree_cent = nx.degree_centrality(G)
    # Betweenness centrality
    betweenness_cent = nx.betweenness_centrality(G)
    # Closeness centrality
    closeness_cent = nx.closeness_centrality(G)
    # Eigenvector centrality
    eigenvector_cent = nx.eigenvector_centrality(G, max_iter=1000)


    # Get the top 10 genes for each measure
    top_degree = sorted(degree_cent.items(), key=lambda x: x[1], reverse=True)[:10]
    top_betweenness = sorted(betweenness_cent.items(), key=lambda x: x[1], reverse=True)[:10]
    top_closeness = sorted(closeness_cent.items(), key=lambda x: x[1], reverse=True)[:10]
    top_eigenvector = sorted(eigenvector_cent.items(), key=lambda x: x[1], reverse=True)[:10]

    print("Top 10 Degree Centrality:", top_degree)
    print("Top 10 Betweenness Centrality:", top_betweenness)
    print("Top 10 Closeness Centrality:", top_closeness)
    print("Top 10 Eigenvector Centrality:", top_eigenvector)

    return G
