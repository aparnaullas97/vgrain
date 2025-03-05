install.packages("VennDiagram")
install.packages("tidyverse")       
install.packages("janitor") 
install.packages("pheatmap") 

library(VennDiagram)
library(tidyverse)
library(janitor)
library(UpSetR)


##################
# METASCAPE
##################
metascape_file <- "Metascape_GO_AllLists.csv"
metascape_data <- read_csv(metascape_file) %>% clean_names()
colnames(metascape_data)
top_processes <- metascape_data %>%
  arrange(desc(log_p)) %>%   
  select(description, log_p) %>%  
  head(n = 20)  
print(top_processes)
macrophage_terms <- c("inflammatory", "cytokine", "immune", "phagocytosis", "macrophage")
top_macrophage_processes <- metascape_data %>%
  filter(str_detect(description, regex(paste(macrophage_terms, collapse = "|"), ignore_case = TRUE))) %>%
  arrange(desc(log_p)) %>%
  select(description, log_p)

print(top_macrophage_processes)
write_csv(top_processes, "top_macrophage_processes.csv")




##################
# STRING
##################





string_file <- "STRING_enrichment.Process.tsv"
string_data <- read_tsv(string_file) %>% clean_names()
print(colnames(string_data))
top_string_processes <- string_data %>%
  arrange(false_discovery_rate) %>%      
  select(term_description, false_discovery_rate, strength) %>%  
  head(n = 20)                           
print(top_string_processes)
top_macrophage_processes_string <- string_data %>%
  filter(str_detect(term_description, regex(paste(macrophage_terms, collapse = "|"), ignore_case = TRUE))) %>%
  arrange(desc(false_discovery_rate)) %>%
  select(term_description, false_discovery_rate)

print(top_macrophage_processes_string)
write_csv(top_string_processes, "top_macrophage_processes_string.csv")



####################
# gProfiler
####################
gprofiler_file <- "gProfiler_hsapiens_2-28-2025_8-39-12 PM__intersections.csv"
gprofiler_data <- read_csv(gprofiler_file) %>% clean_names()
print(colnames(gprofiler_data))
macrophage_gprofiler <- gprofiler_data %>%
  filter(str_detect(term_name, regex(paste(macrophage_terms, collapse = "|"), ignore_case = TRUE))) %>%
  arrange(adjusted_p_value) %>%  # Lower adjusted p-value indicates higher significance
  select(term_name, adjusted_p_value, negative_log10_of_adjusted_p_value)
print(macrophage_gprofiler)
write_csv(macrophage_gprofiler, "macrophage_gprofiler.csv")




metascape_terms <- top_macrophage_processes$description 
gprofiler_terms <- macrophage_gprofiler$term_name
string_terms <- top_macrophage_processes_string$term_description  

clean_terms <- function(terms) {
  terms %>% 
    tolower() %>% 
    trimws() %>% 
    unique()
}

metascape_terms <- clean_terms(metascape_terms)
gprofiler_terms <- clean_terms(gprofiler_terms)
string_terms <- clean_terms(string_terms)


common_mg <- intersect(metascape_terms, gprofiler_terms)
common_ms <- intersect(metascape_terms, string_terms)
common_gs <- intersect(gprofiler_terms, string_terms)

common_all <- Reduce(intersect, list(metascape_terms, gprofiler_terms, string_terms))

cat("Common between Metascape and g:Profiler:", common_mg, "\n")
cat("Common between Metascape and STRING:", common_ms, "\n")
cat("Common between g:Profiler and STRING:", common_gs, "\n")
cat("Common to all three tools:", common_all, "\n")

