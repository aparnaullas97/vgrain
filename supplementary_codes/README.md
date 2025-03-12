# Data simulation and Downstream analysis scripts supporting VGRAIN Model
This folder contains grn2gex_scMultiSim.R script which clusters gene regulatory networks (GRNs) and simulates gene expression data using the 
[grn2gex](https://github.com/AnneHartebrodt/grn2gex/tree/main) and [scMultiSim](https://github.com/ZhangLabGT/scMultiSim) R packages respectively. 
The generated data is structured for further analysis in the V-GRAIN model.
This repository also contains Enrichment.R script for analyzing pathway enrichment data from **Metascape, STRING, and g:Profiler**.

## Dependencies
Before running the scripts, install the required R packages:

```r
install.packages("igraph")
install.packages("grn2gex")
install.packages("scMultiSim")
install.packages("data.table")
install.packages("ggplot2")
install.packages("tidyr")
install.packages("VennDiagram")
install.packages("tidyverse")
install.packages("janitor")
install.packages("pheatmap")
```
Load the libraries in your R environment:
```r
library(igraph)
library(grn2gex)
library(scMultiSim)
library(data.table)
library(ggplot2)
library(tidyr)
library(VennDiagram)
library(tidyverse)
library(janitor)
library(UpSetR)
```

## grn2gex_scMultiSim R Script
1. **GRN Simulation (Preprocessing for VGRAIN)**
   - Reads a network from `example_network.tsv`.
   - Clusters the GRN using `grn2gex`.
   - Selects a specific cluster.
   - Simulates **gene expression data** using `scMultiSim`.
   - Saves expression data and GRN edges as CSV files for GNN analysis.
2. **Batch Processing of GRN Clusters**
   - Iterates over Clusters 1 to number of clusters.
   - Simulates gene expression data for each cluster.
   - Saves the network and expression data as CSV files.

## Enrichment R Script
1. **Metascape Analysis**
   - Reads `Metascape_GO_AllLists.csv`.
   - Extracts top 20 pathways.
   - Filters macrophage-related processes.
   - Saves output as `top_macrophage_processes.csv`.

2. **STRING Analysis**
   - Reads `STRING_enrichment.Process.tsv`.
   - Extracts top 20 pathways based on **False Discovery Rate (FDR)**.
   - Filters macrophage-related processes.
   - Saves output as `top_macrophage_processes_string.csv`.

3. **g:Profiler Analysis**
   - Reads `gProfiler_hsapiens_*.csv`.
   - Filters macrophage-related pathways.
   - Saves output as `macrophage_gprofiler.csv`.
    
4. **Common Pathway Analysis**
   - Compares pathway enrichment results from Metascape, STRING, and g:Profiler.
   - Identifies overlapping pathways.
   - Outputs common pathways to console.

## How to Run
1. Place all input files (`.csv` and `.tsv`) in the working directory.
2. Open an R session and set the working directory:
   ```r
   setwd("/path/to/your/data")
   ```
3. Run the preprocessing script:
   ```r
   source("grn2gex_scMultiSim.R")
   ```
4. Train V-GRAIN on the simulated datasets
5. Submit the results in downstream analysis platforms ([Metascape](https://metascape.org/gp/index.html#/main/step1), [g:Profiler](https://biit.cs.ut.ee/gprofiler/gost), [STRING](https://string-db.org/cgi/input?sessionId=byAkilCDAjQa&input_page_active_form=multiple_identifiers))
4. Run the Enrichment script with the results from downstream analysis:
   ```r
   source("Enrichment.R")
   ```

## Outputs
- **Simulated gene expression data** for different GRN clusters (`expr_data.csv`, `gex_clusterX.csv`).
- **GRN edges** for GNN analysis (`net_clusterX.csv`).
- **Filtered pathway lists** (`.csv` files) for each tool.
