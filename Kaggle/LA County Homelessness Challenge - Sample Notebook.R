# ==============================================================================
# PREDICTIVE MODELING: SOCIOECONOMIC DETERMINANTS OF HOMELESSNESS 
# IN LOS ANGELES, CA
# ==============================================================================
# Author: Elgas
# Date: 2024-03-12
# 
# RESEARCH OBJECTIVE:
# Determine which socioeconomic features have the greatest predictive power
# for homelessness rates across Los Angeles County.
#
# METHODOLOGY:
# 1. Process 2020 Census socioeconomic indicators at the census tract level
# 2. Aggregate data to SPA level for analysis (8 SPAs total)
# 3. Calculate homelessness rates using 2020 Point-in-Time count data
# 4. Build predictive models to identify key drivers of homelessness
# 5. Quantify feature importance and impact on homelessness outcomes
#
# DATA PREPARATION STEPS:
# 1. Create geographic lookup tables linking ZIP codes, Census tracts, and SPAs
# 2. Process multiple Census tables for demographic and socioeconomic indicators
# 3. Validate data consistency across Census tables
# 4. Calculate homelessness rates by SPA as the target variable
# ==============================================================================

# ==============================================================================
# SETUP AND ENVIRONMENT
# ==============================================================================

# Clear workspace
rm(list = ls()) 

# Load required libraries
library(ggplot2)      # Data visualization
library(dplyr)        # Data manipulation
library(stringr)      # String operations
library(psych)        # Psychological research tools
library(tidyr)        # Data tidying
library(car)          # Companion to Applied Regression
library(corrplot)     # Correlation plots
library(betareg)      # Beta regression models
library(lmtest)       # Linear model tests
library(purrr)        # Functional programming tools
library(xgboost)      # Extreme gradient boosting
library(Hmisc)        # Miscellaneous functions
library(data.table)   # Enhanced data frame
library(caret)        # Classification and regression training
library(randomForest) # Random forest algorithm

print("Environment setup complete")

# ==============================================================================
# CREATE GEOGRAPHIC LOOKUP TABLES
# ==============================================================================

# Load SPA-ZIP code mapping
# Source: California Fund for Health and Human Services
# URL: https://connect.calfund.org/oldsite/document.doc?id=1047
df_spa <- read.csv("datasets/SPA/SPA_ZIP_Calfund.csv", 
                   sep = ",", header = TRUE, skip = 0)

# Standardize column name for ZIP codes
names(df_spa)[3] <- "ZIP"

# Load Census Tract to ZIP code mapping
# Source: U.S. Census Bureau 2020 ZCTA to Tract Relationship File
# URL: https://www.census.gov/programs-surveys/geography/technical-documentation/records-layout/2020-zcta-record-layout.html
df_zip_tract <- read.csv("datasets/lookup/tab20_zcta520_tract20_natl.csv", 
                         sep = ",", header = TRUE, skip = 0)

# Standardize ZIP code column name
names(df_zip_tract)[names(df_zip_tract) == "GEOID_ZCTA5_20"] <- "ZIP"

# Create comprehensive SPA-ZIP-Tract lookup table
df_spa_zip_tract <- merge(df_spa, 
                          df_zip_tract[c("GEOI", "ZIP", "TRACT", "NAMELSAD_TRACT_20")],
                          by = "ZIP", all.x = TRUE)

# Select and reorder relevant columns
df_spa_zip_tract <- df_spa_zip_tract[c("SPA", "Communities", "ZIP", "TRACT", 
                                       "GEOI", "NAMELSAD_TRACT_20")]

# Save lookup table for future use
write.csv(df_spa_zip_tract, "datasets/lookup/df_spa_zip_tract.csv", row.names = FALSE)

print(paste("Created lookup table with", nrow(df_spa_zip_tract), "SPA-Tract mappings"))

# ==============================================================================
# CENSUS DATA PROCESSING FUNCTIONS
# ==============================================================================

#' Load and Process Census Tables
#' 
#' This function standardizes the processing of raw Census CSV files by:
#' - Cleaning headers and extracting tract numbers
#' - Converting data types appropriately
#' - Adding SPA geographic identifiers
#' - Saving processed files for future use
#'
#' @param file_path String. Path to the raw Census CSV file
#' @param table_name String. Identifier for the table (e.g., "p1", "b18107")
#' @return data.frame. Processed Census table with SPA column added
#' @note Handles both Decennial Census (" !!Total:") and ACS ("Estimate!!Total:") formats
load_census_table <- function(file_path, table_name) {
  
  cat("Processing", table_name, "from", basename(file_path), "\n")
  
  # Read raw CSV file without assuming header structure
  df <- read.csv(file_path, sep = ",", header = FALSE)
  
  # Remove metadata row and set proper column names
  df <- df[-1, ]  # Remove first row (typically GEO_ID, NAME, etc.)
  names(df) <- as.character(df[1, ])  # Use second row as column names
  df <- df[-1, ]  # Remove the header row
  
  # Remove geography column (first column contains full geographic names)
  df <- df[, -1]
  
  # Extract clean tract numbers from geographic names
  # Pattern: "Census Tract 1234.56, Los Angeles County, California"
  df[, 1] <- gsub("Census Tract ([0-9.]+),.*", "\\1", df[, 1])
  names(df)[1] <- "TRACT"
  
  # Remove decimal points and convert to integer for joining
  df$TRACT <- gsub("\\.", "", df$TRACT)
  df$TRACT <- as.integer(df$TRACT)
  
  # Identify and process the total population/count column
  if (" !!Total:" %in% colnames(df)) {
    total_col <- " !!Total:"  # Decennial Census format
  } else if ("Estimate!!Total:" %in% colnames(df)) {
    total_col <- "Estimate!!Total:"  # American Community Survey format
  } else {
    stop(paste("No Total column found in", table_name))
  }
  
  # Clean and convert total column to integer
  # Replace missing values with 0 for aggregation purposes
  df[[total_col]][is.na(df[[total_col]]) | is.null(df[[total_col]])] <- 0
  df[[total_col]] <- as.integer(df[[total_col]])
  
  # Add SPA geographic identifiers through tract mapping
  df <- merge(df, df_spa_zip_tract[, c("TRACT", "SPA")], 
              by = "TRACT", all.x = TRUE)
  
  # Reorder columns to put SPA first for easier analysis
  df <- df[, c("SPA", setdiff(names(df), "SPA"))]
  
  # Save processed table
  output_path <- paste0("datasets/census/df_census_", tolower(table_name), ".csv")
  write.csv(df, output_path, row.names = FALSE)
  
  cat("Saved processed table to", output_path, "\n")
  cat("Table dimensions:", nrow(df), "rows x", ncol(df), "columns\n")
  
  return(df)
}

#' Calculate SPA Population Totals
#' 
#' Aggregates population counts by Service Planning Area from Census tables
#'
#' @param census_df data.frame. Census table with SPA column
#' @param table_name String. Name for the output column
#' @return data.frame. SPA totals with renamed column
get_spa_totals <- function(census_df, table_name) {
  
  # Remove tracts that don't map to any SPA
  valid_data <- census_df[!is.na(census_df$SPA), ]
  
  # Identify the correct total column
  if ("Estimate!!Total:" %in% colnames(valid_data)) {
    total_col <- "Estimate!!Total:"
  } else if (" !!Total:" %in% colnames(valid_data)) {
    total_col <- " !!Total:"
  } else {
    stop("No Total column found")
  }
  
  # Aggregate by SPA
  totals <- aggregate(valid_data[total_col], 
                      by = list(SPA = valid_data$SPA), 
                      FUN = sum, na.rm = TRUE)
  
  # Rename the total column for clarity
  colnames(totals)[2] <- table_name
  
  return(totals)
}

# ==============================================================================
# LOAD AND PROCESS CENSUS TABLES
# ==============================================================================

# Load five key Census tables for socioeconomic analysis:

# P1: Total Population (2020 Decennial Census)
# - Most recent official population count
# - Used as denominator for calculating rates
df_census_p1 <- load_census_table("datasets/census/DECENNIALPL2020.P1-Data.csv", "p1")

# B18107: Age by Disability Status (2020 5-Year ACS)
# - Provides disability prevalence data
# - Important socioeconomic vulnerability indicator  
df_census_b18107 <- load_census_table("datasets/census/ACSDT5Y2020.B18107-Data.csv", "b18107")

# B23025: Employment Status (2020 5-Year ACS)
# - Labor force participation and unemployment
# - Key economic indicator for homelessness risk
df_census_b23025 <- load_census_table("datasets/census/ACSDT5Y2020.B23025-Data.csv", "b23025")

# B25014: Tenure by Occupancy Status (2020 5-Year ACS)
# - Housing occupancy and ownership patterns
# - Housing stability indicator
df_census_b25014 <- load_census_table("datasets/census/ACSDT5Y2020.B25014-Data.csv", "b25014")

# B25070: Gross Rent as Percentage of Income (2020 5-Year ACS)
# - Housing cost burden indicator
# - Critical factor in housing stability and homelessness risk
df_census_b25070 <- load_census_table("datasets/census/ACSDT5Y2020.B25070-Data.csv", "b25070")

# ==============================================================================
# DATA VALIDATION: COMPARE POPULATION TOTALS ACROSS TABLES
# ==============================================================================

cat("\n=== DATA VALIDATION ===\n")
cat("Comparing population totals across Census tables by SPA\n")
cat("Note: P1 should have highest totals (total population)\n")
cat("ACS tables represent subsets or different universes\n\n")

# Calculate SPA totals for each table
spa_totals_P1 <- get_spa_totals(df_census_p1, "Table_P1")
spa_totals_B18107 <- get_spa_totals(df_census_b18107, "Table_B18107")
spa_totals_B23025 <- get_spa_totals(df_census_b23025, "Table_B23025")
spa_totals_B25014 <- get_spa_totals(df_census_b25014, "Table_B25014")
spa_totals_B25070 <- get_spa_totals(df_census_b25070, "Table_B25070")

# Combine all totals for comparison
population_comparison <- merge(spa_totals_P1, spa_totals_B18107, by = "SPA", all = TRUE)
population_comparison <- merge(population_comparison, spa_totals_B23025, by = "SPA", all = TRUE)
population_comparison <- merge(population_comparison, spa_totals_B25014, by = "SPA", all = TRUE)
population_comparison <- merge(population_comparison, spa_totals_B25070, by = "SPA", all = TRUE)

print("Population totals by SPA across all Census tables:")
print(population_comparison)

# ==============================================================================
# CALCULATE BASELINE SPA POPULATIONS
# ==============================================================================

cat("\n=== BASELINE POPULATION CALCULATION ===\n")

# Use P1 (Decennial Census) as the authoritative population count
# This provides the most accurate population denominators for rate calculations
merged_data <- merge(df_spa_zip_tract[, c("SPA", "TRACT")],
                     df_census_p1[, c("TRACT", " !!Total:")],
                     by = "TRACT")

# Aggregate to SPA level
df_spa_population <- aggregate(merged_data[" !!Total:"],
                               by = list(SPA = merged_data$SPA),
                               FUN = sum, na.rm = TRUE)

# Clean column names
colnames(df_spa_population)[2] <- "Total_Population"

print("SPA Population totals (from Census P1):")
print(df_spa_population)

# ==============================================================================
# CALCULATE HOMELESSNESS RATES BY SPA
# ==============================================================================

cat("\n=== HOMELESSNESS RATE CALCULATION ===\n")

# Load 2020 Point-in-Time (PIT) Count data
# This represents the official count of homeless individuals on a single night
df_pit <- read.csv("datasets/SPA/input_2020_PIT.csv", sep = ",", check.names = FALSE)

cat("PIT data structure:\n")
print(head(df_pit))

# Filter for person-level counts (exclude household counts for individual analysis)
filtered_data <- df_pit[df_pit$Feature %in% c("All Persons", "All Households"), ]

# Aggregate homeless counts by SPA and category
df_spa_summary <- aggregate(cbind(Sheltered, Unsheltered, Total) ~ SPA + Feature,
                            data = filtered_data,
                            FUN = sum, na.rm = TRUE)

cat("\nHomeless counts by SPA:\n")
print(df_spa_summary)

# Focus on individual persons for rate calculation
all_persons <- df_spa_summary[df_spa_summary$Feature == "All Persons", ]

# Combine homeless counts with population data
df_spa_ratio <- merge(all_persons[, c("SPA", "Total")],
                      df_spa_population,
                      by = "SPA")

# Calculate homelessness rate (homeless persons per total population)
df_spa_ratio$Homelessness_Rate <- df_spa_ratio$Total / df_spa_ratio$Total_Population

# Clean up column names for clarity
colnames(df_spa_ratio)[colnames(df_spa_ratio) == "Total"] <- "Homeless_Persons"

# Select final columns
df_spa_ratio <- df_spa_ratio[, c("SPA", "Homeless_Persons", "Total_Population", "Homelessness_Rate")]

cat("\nFinal homelessness rates by SPA:\n")
print(df_spa_ratio)

cat("\n=== DATA PROCESSING COMPLETE ===\n")
cat("Processed datasets available for analysis:\n")
cat("- df_spa_zip_tract: Geographic lookup table\n")
cat("- df_census_[table]: Five Census tables with socioeconomic indicators\n") 
cat("- df_spa_ratio: Homelessness rates by SPA\n")
cat("\nData is ready for feature engineering and modeling.\n")


