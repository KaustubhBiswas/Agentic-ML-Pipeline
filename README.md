# Agentic ML Pipeline

This project provides an agent-based pipeline for semantic classification and parsing of tabular data columns, such as phone numbers, company names, countries, and dates. It combines machine learning-based column type prediction with specialized parsing logic for normalization and extraction.

## Features

- **Column Type Classification:** Uses a trained ML model to predict the semantic type of each column in a CSV file (e.g., Phone Number, Company Name, Country, Date, Other).
- **Parsing & Normalization:** Extracts and normalizes phone numbers and company names using custom logic and external resources.
- **Confidence Thresholding:** Flags columns with low classification confidence for manual review.
- **Extensible:** Easily add new types or parsing logic.

## File Overview

- [`agent.py`](agent.py): Main orchestrator that combines prediction and parsing. Processes input CSVs and outputs structured results.
- [`predict.py`](predict.py): Contains ML model training, feature extraction, and column type classification logic.
- [`parser.py`](parser.py): Provides functions to parse and normalize phone numbers and company names.
- `phone.csv`, `company.csv`, `countries.txt`, `dates.csv`, `legal.txt`: Training and reference data files (required for model training and parsing).
- `README.md`: Project documentation.

## Installation

1. **Clone the repository** and navigate to the project directory.
2. **Install dependencies:**
   ```sh
   pip install pandas numpy scikit-learn xgboost phonenumbers python-dateutil

## Testing

    **Run the following command from the root directory of the project**
    ```sh
    python agent.py --input TrainingData