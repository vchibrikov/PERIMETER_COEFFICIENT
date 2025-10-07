# -----------------------------------------------------------------------------
#
# Script: Perimeter Coefficient Analysis
# Author: Vadym Chibrikov
# Date: 2025-10-07
#
# Description: This script processes object dimension data (x and y axis lengths)
#              to calculate a custom "perimeter coefficient", along with grid
#              width, length, and elongation. It performs statistical analysis
#              (ANOVA + Tukey's HSD) and generates a 2x2 summary figure.
#
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# SECTION 1: SETUP AND CONFIGURATION
# -----------------------------------------------------------------------------

# --- 1.1: Load Required Libraries ---
if (!require("pacman")) install.packages("pacman")
pacman::p_load(
  'tidyverse', 'readxl', 'writexl', 'agricolae', 'Cairo', 'bbplot', 'gridExtra'
)

# --- 1.2: Configuration ---
# --- MERGE RAW DATA (OPTIONAL) ---
# Set RUN_MERGE to TRUE to merge all .xlsx files in the specified folder first.
RUN_MERGE <- FALSE
MERGE_INPUT_DIR <- "./merge/input/directory"
MERGED_DATA_PATH <- "./merged/data/file/path.xlsx"

# --- MAIN ANALYSIS PATHS ---
DATA_PATH <- "./merged/data/file/path.xlsx"
SCALE_DATA_PATH <- "./scale/data/path.xlsx"
OUTPUT_DIR <- "./output/directory/path"

# --- ANALYSIS PARAMETERS ---
SAMPLES_TO_ANALYZE <- c("SAMPLES", "TO", "ANALYZE") # Define sample groups and their plot order.

# --- 1.3: Plotting Parameters ---
colors <- RColorBrewer::brewer.pal(9, "Set1")


# -----------------------------------------------------------------------------
# SECTION 2: HELPER FUNCTIONS
# -----------------------------------------------------------------------------

#' Merge multiple XLSX files into a single file.
merge_xlsx_files <- function(input_dir, output_file) {
  file_paths <- list.files(path = input_dir, pattern = "*.xlsx", full.names = TRUE)
  if (length(file_paths) == 0) {
    stop("No .xlsx files found in the specified directory.")
  }
  
  # Corrected function call for map_dfr
  merged_data <- map_dfr(file_paths, read_xlsx)
  
  write_xlsx(merged_data, output_file)
  print(paste("Merged", length(file_paths), "files into", output_file))
}


#' Perform ANOVA and HSD test, then summarize data.
perform_analysis <- function(data, metric_var, group_var = "file_group") {
  formula <- as.formula(paste(metric_var, "~", group_var))
  anova_model <- aov(formula, data = data)
  hsd_test <- HSD.test(anova_model, group_var, group = TRUE, console = FALSE)
  hsd_groups <- as_tibble(hsd_test$groups, rownames = group_var)

  data %>%
    group_by(.data[[group_var]]) %>%
    summarise(
      mean = mean(.data[[metric_var]], na.rm = TRUE),
      sd = sd(.data[[metric_var]], na.rm = TRUE),
      .groups = 'drop'
    ) %>%
    left_join(hsd_groups %>% select(all_of(group_var), groups), by = group_var)
}

#' Create a standard summary bar plot.
create_summary_plot <- function(summary_data, y_label, y_limits, y_intercept) {
  ggplot(summary_data, aes(x = file_group, y = mean, color = file_group)) +
    geom_bar(stat = "identity", fill = NA, linewidth = 1) +
    geom_errorbar(aes(ymin = mean - sd, ymax = mean + sd), width = 0.2, linewidth = 1) +
    geom_text(aes(label = groups, y = mean + sd), vjust = -0.7, color = "black", size = 6) +
    geom_hline(yintercept = y_intercept, linetype = "dashed", color = "#333333") +
    scale_y_continuous(limits = y_limits) +
    scale_color_manual(values = colors) +
    labs(x = "Sample", y = y_label) +
    bbc_style() +
    theme(
      aspect.ratio = 1,
      legend.position = "none",
      axis.title = element_text(size = 18, face = "bold"),
      axis.text.y = element_text(size = 16),
      axis.text.x = element_text(size = 14, angle = 0)
    )
}

# -----------------------------------------------------------------------------
# SECTION 3: DATA LOADING AND PREPARATION
# -----------------------------------------------------------------------------

# --- 3.1: Merge Raw Data Files (if configured) ---
if (RUN_MERGE) {
  merge_xlsx_files(MERGE_INPUT_DIR, MERGED_DATA_PATH)
}

# --- 3.2: Load and Prepare Master Dataset ---
# Load main data and scale data once, then join them.
raw_data <- read_excel(DATA_PATH)
scale_data <- read_excel(SCALE_DATA_PATH)

# Prepare and join data
data <- raw_data %>%
  na.omit() %>%
  mutate(date = str_extract(filename, "\\d{4}_\\d{2}_\\d{2}_\\d{2}")) %>%
  left_join(
    scale_data %>% mutate(date = str_extract(filename, "\\d{4}_\\d{2}_\\d{2}_\\d{2}")),
    by = "date"
  ) %>%
  select(
    filename = filename.x,
    y_axis_length_px,
    x_axis_length_px,
    distance_px_per_30_mm
  ) %>%
  na.omit()

# --- 3.3: Calculate Metrics ---
# All metrics are calculated in a single, grouped operation.
analysis_data <- data %>%
  tidyr::extract(
    filename,
    into = c("composition_nr", "pressure_kpa", "speed_mm_min", "grid_nr"),
    regex = ".*_(\\d{2})_(\\d{2})_(\\d{2})_(\\d+).*",
    remove = FALSE
  ) %>%
  mutate(filename_group = paste(composition_nr, pressure_kpa, speed_mm_min, grid_nr, sep = "_")) %>%
  group_by(filename_group) %>%
  summarise(
    # Intermediate values for perimeter coefficient
    value_1 = (mean(x_axis_length_px, na.rm = TRUE) + mean(y_axis_length_px, na.rm = TRUE)) / mean(distance_px_per_30_mm),
    value_2 = (sd(x_axis_length_px, na.rm = TRUE) + sd(y_axis_length_px, na.rm = TRUE)) / mean(distance_px_per_30_mm),
    
    # Grid dimensions in mm
    width_mm = (mean(y_axis_length_px, na.rm = TRUE) / mean(distance_px_per_30_mm)) * 30,
    length_mm = (mean(x_axis_length_px, na.rm = TRUE) / mean(distance_px_per_30_mm)) * 30,
    
    .groups = 'drop'
  ) %>%
  mutate(
    # Final calculation for perimeter coefficient
    # Formula: 1 / (((0.5 * V1 - 1) + 1) * (1 + 0.5 * V2))
    perimeter_coef = 1 / (((0.5 * value_1 - 1) + 1) * (1 + 0.5 * value_2)),
    
    # Final calculation for elongation
    elongation_pct = (abs(width_mm - length_mm) / 30) * 100
  ) %>%
  # Extract sample group info for final analysis
  tidyr::extract(
    filename_group,
    into = c("composition_nr", "pressure_kpa", "speed_mm_min"),
    regex = "(\\d+)_(\\d+)_(\\d+)_.*",
    remove = FALSE
  ) %>%
  mutate(file_group = paste(composition_nr, pressure_kpa, speed_mm_min, sep = "_")) %>%
  # Filter for desired samples and set factor order
  filter(file_group %in% SAMPLES_TO_ANALYZE) %>%
  mutate(file_group = factor(file_group, levels = SAMPLES_TO_ANALYZE))


# -----------------------------------------------------------------------------
# SECTION 4: PERFORM ANALYSIS AND GENERATE PLOTS
# -----------------------------------------------------------------------------

# --- 4.1: Perimeter Coefficient ---
summary_coeff <- perform_analysis(analysis_data, "perimeter_coef")
plot_coeff <- create_summary_plot(summary_coeff, "Perimeter coefficient (a.u.)", c(0, 1.25), 1)

# --- 4.2: Grid Width (mm) ---
summary_width <- perform_analysis(analysis_data, "width_mm")
plot_width <- create_summary_plot(summary_width, "Grid width (mm)", c(0, 40), 30)

# --- 4.3: Grid Length (mm) ---
summary_length <- perform_analysis(analysis_data, "length_mm")
plot_length <- create_summary_plot(summary_length, "Grid length (mm)", c(0, 40), 30)

# --- 4.4: Grid Elongation (%) ---
summary_elong <- perform_analysis(analysis_data, "elongation_pct")
plot_elong <- create_summary_plot(summary_elong, "Grid elongation (%)", c(0, 3), 0)


# -----------------------------------------------------------------------------
# SECTION 5: ASSEMBLE AND SAVE FINAL FIGURE
# -----------------------------------------------------------------------------

dir.create(OUTPUT_DIR, showWarnings = FALSE, recursive = TRUE)
summary_filename <- file.path(OUTPUT_DIR, paste0("PERIMETER_COEFFICIENT_SUMMARY_", SAMPLES_TO_ANALYZE[1], ".jpeg"))

CairoJPEG(filename = summary_filename, width = 30, height = 30, units = "cm", dpi = 600, bg = "white")

grid.arrange(
  plot_length,
  plot_width,
  plot_coeff,
  plot_elong,
  layout_matrix = rbind(
    c(1, 2),
    c(3, 4)
  )
)

dev.off()

print(paste("Analysis complete. Summary figure saved to:", summary_filename))
