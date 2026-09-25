# Module 1 — Data Pipeline

## Overview

This module implements an end-to-end data engineering pipeline using Books to Scrape.

The pipeline includes:
1. Web scraping
2. Data cleaning
3. Currency conversion
4. SQLite database creation
5. SQL analysis
6. Pandas SQL and merge validation

## Data Source

Source: https://books.toscrape.com/

Three categories were scraped:
- Fiction
- Mystery
- Historical Fiction

The final dataset contains 123 books across 3 categories.

## Fields Scraped

- title
- price
- star_rating
- availability
- category

## Data Cleaning

The scraped price was converted into a numeric `price_gbp` column.

Star ratings were converted as:
- One = 1
- Two = 2
- Three = 3
- Four = 4
- Five = 5

Availability was converted into the Boolean `in_stock` column.

If a numeric parsing failure occurs, median imputation is used to prevent
unexpected values from stopping the pipeline.

## Currency Conversion

The project-defined fixed conversion rate is:

**1 GBP = 105.50 INR**

Formula:

`price_inr = price_gbp * 105.50`

No live currency API is required.

## Database Design

The SQLite database contains two normalized tables.

### categories

- category_id — Primary Key
- category_name

### books

- book_id — Primary Key
- title
- price_gbp
- price_inr
- rating
- in_stock
- category_id — Foreign Key

The `category_id` field creates the relationship between books and categories.

## SQL Analysis

The project demonstrates:
- SELECT
- WHERE
- ORDER BY
- LIMIT
- DISTINCT
- BETWEEN
- IN
- INNER JOIN

Six SQL queries and their outputs are saved in `query_outputs.txt`.

## Pandas Validation

Two SQL query results are loaded using `pd.read_sql()`.

The SQL JOIN is also recreated using `pd.merge()`.

The two outputs were compared and confirmed to be equivalent.

Result:

`SQL JOIN == pd.merge(): True`

## Files

- module1_data_pipeline.ipynb
- books_cleaned.csv
- books.db
- query_outputs.txt
- README.md
- requirements.txt

## How to Run

Install the required packages:

`pip install requests beautifulsoup4 pandas`

Then run all cells in `module1_data_pipeline.ipynb` from top to bottom.
