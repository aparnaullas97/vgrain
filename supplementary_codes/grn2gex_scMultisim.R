###############################
# 1. Load Required Libraries
###############################
library(igraph)
library(grn2gex)
library(scMultiSim)
library(data.table)
library(ggplot2)
library(tidyr)

###############################
# 2. Set Paths and Load Data
###############################
net.dir <- "path/to/simulation/collectri_subnetworks"
collectri <- "path/to/collecTRI/collectri_network.tsv"
gex.dir <- "path/to/simulation/collectri_simulated_data"

# Ensure the output directory exists
dir.create(gex.dir, recursive = TRUE, showWarnings = FALSE)

# Load CoLlecTRI network data
collectri_data <- fread(collectri, sep = "\t", header = TRUE)
if (!all(colnames(collectri_data) %in% c("source", "target"))) {
  stop("⚠️ Column names in CoLlecTRI file are incorrect. They should be 'source' and 'target'.")
}

###############################
# 3. Cluster Networks using grn2gex
###############################
clustered_networks <- clusterNetwork(collectri_data, min_nodes = 50, max_nodes = 200)
if (!is.list(clustered_networks) || length(clustered_networks) == 0) {
  stop("⚠️ No clustered networks were generated!")
}
cat("✅ Successfully clustered", length(clustered_networks), "networks\n")

selected_cluster <- clustered_networks[[12]]

str(selected_cluster)

if (inherits(selected_cluster, "igraph")) {
  cluster_edges <- as_data_frame(selected_cluster, what = "edges")
} else if (is.data.frame(selected_cluster)) {
  cluster_edges <- selected_cluster
} else {
  stop("Selected cluster is not in an expected format.")
}

if (!all(c("source", "target") %in% colnames(cluster_edges))) {
  if (all(c("from", "to") %in% colnames(cluster_edges))) {
    names(cluster_edges)[names(cluster_edges) == "from"] <- "source"
    names(cluster_edges)[names(cluster_edges) == "to"] <- "target"
  } else {
    stop("The edge list does not have recognizable 'source' and 'target' columns.")
  }
}

if (!"effect" %in% colnames(cluster_edges)) {
  cluster_edges$effect <- 1
}
grn_for_sim <- data.frame(
  target = cluster_edges$target,
  regulator = cluster_edges$source,
  effect = cluster_edges$effect
)
cat("Dimensions of grn_for_sim:", dim(grn_for_sim), "\n")


###############################
# 4. Simulate Gene Expression Data Using scMultiSim
###############################
set.seed(123)
sim_options <- list(
  GRN = grn_for_sim,       
  num.cells = 1000,        
  num.cif = 20,            
  intrinsic.noise = 0.5    
)
sim_results <- sim_true_counts(sim_options)

counts_df <- as.data.frame(sim_results$counts)
write.csv(counts_df, file = file.path(gex.dir, "expr_data.csv"), row.names = TRUE)

###############################
# 5. Visualize Simulated Gene Expression Data
###############################
counts_long <- pivot_longer(counts_df, cols = everything(), 
                            names_to = "cell", values_to = "expression")
ggplot(counts_long, aes(x = cell, y = expression)) +
  geom_boxplot() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  labs(title = "Simulated Gene Expression Data", x = "Cells", y = "Expression Levels")

###############################
# 6. Visualize the Selected Gene Regulatory Network (GRN)
###############################
grn_graph <- graph_from_data_frame(cluster_edges, directed = TRUE)
plot(grn_graph,
     vertex.size = 5,
     vertex.label.cex = 0.7,
     edge.arrow.size = 0.5,
     main = "GRN from Selected Cluster",
     layout = layout_with_fr)



###############################
# 7. Prepare Data for GNN
###############################
expression_matrix <- as.matrix(counts_df)
node_features <- as.data.frame(expression_matrix)

edge_labels <- data.frame(
  source = cluster_edges$source,
  target = cluster_edges$target,
  label = 1
)

grn_edges_out <- cluster_edges[, c("source", "target")]

colnames(grn_edges_out) <- c("Gene1", "Gene2")

write.csv(grn_edges_out, file = "path/to/save/net_cluster12.csv", row.names = FALSE)

write.csv(counts_df, file = "path/to/save/gex_cluster12.csv", row.names = TRUE)







###############################
# 4. Process Clusters 13 to 43 in a Loop
###############################
for (i in 13:43) {
  
  cat("Processing cluster", i, "\n")
  
  current_cluster <- clustered_networks[[i]]
  
  if (inherits(current_cluster, "igraph")) {
    cluster_edges <- as_data_frame(current_cluster, what = "edges")
  } else if (is.data.frame(current_cluster)) {
    cluster_edges <- current_cluster
  } else {
    warning("Cluster ", i, " is not in an expected format. Skipping.")
    next
  }
  
  if (!all(c("source", "target") %in% colnames(cluster_edges))) {
    if (all(c("from", "to") %in% colnames(cluster_edges))) {
      names(cluster_edges)[names(cluster_edges) == "from"] <- "source"
      names(cluster_edges)[names(cluster_edges) == "to"] <- "target"
    } else {
      warning("Cluster ", i, " does not have recognizable 'source' and 'target' columns. Skipping.")
      next
    }
  }
  
  if (!"effect" %in% colnames(cluster_edges)) {
    cluster_edges$effect <- 1
  }
  
  grn_for_sim <- data.frame(
    target = cluster_edges$target,
    regulator = cluster_edges$source,
    effect = cluster_edges$effect
  )
  
  cat("Dimensions of grn_for_sim for cluster", i, ":", dim(grn_for_sim), "\n")
  
  ###############################
  # 5. Simulate Gene Expression Data Using scMultiSim
  ###############################
  set.seed(123)
  sim_options <- list(
    GRN = grn_for_sim,       
    num.cells = 1000,        
    num.cif = 20,            
    intrinsic.noise = 0.5    
  )
  
  sim_results <- sim_true_counts(sim_options)
  counts_df <- as.data.frame(sim_results$counts)
  
  gex_file <- paste0("path/to/gex_cluster", i, ".csv")
  write.csv(counts_df, file = gex_file, row.names = TRUE)
  
  ###############################
  # 6. Prepare and Save GRN Edges File for GNN
  ###############################
  grn_edges_out <- cluster_edges[, c("source", "target")]
  colnames(grn_edges_out) <- c("Gene1", "Gene2")
  
  net_file <- paste0("path/to/net_cluster", i, ".csv")
  write.csv(grn_edges_out, file = net_file, row.names = FALSE)
  
  cat("Finished processing cluster", i, "\n\n")
}
