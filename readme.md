# Instacart Association Rule Mining

This repository contains Python code for performing association rule mining on transaction data from Instacart. Association rule mining is a technique used to discover interesting relationships between items in a dataset.

## Dependencies

Before running the code, make sure you have the following libraries installed:

- pandas
- mlxtend

You can install these libraries using pip:

```bash
pip install pandas mlxtend
```

## Usage

1. Clone this repository:

```bash
git clone https://github.com/your-username/instacart-association-rules.git
```

2. Navigate to the repository directory:

```bash
cd instacart-association-rules
```

3. Place your transaction data in a CSV file named "InstacartTransactions.csv" with columns for OrderID and ItemID.

4. Run the Python script:

```bash
python instacart_association_rules.py
```

## Overview

The provided Python script analyzes transaction data from Instacart to discover association rules between items. The code performs the following steps:

1. Loads the transaction data from "InstacartTransactions.csv" into a pandas DataFrame.

2. Converts the transaction data into a one-hot encoded format, where each row represents a transaction, and each column represents an item with binary values indicating whether the item was purchased.

3. Calculates item frequencies, indicating how often each item is purchased.

4. Uses the Apriori algorithm to find frequent item sets based on a minimum support threshold.

5. Generates association rules with confidence and lift metrics, filtering based on a minimum confidence threshold.

6. Sorts and displays the top association rules based on lift.

Make sure to adjust the file paths and column names in the script to match your dataset if needed.

## Dataset

The dataset is expected to be in a CSV file named "InstacartTransactions.csv" with at least two columns:

- OrderID: A unique identifier for each transaction.
- ItemID: A unique identifier for each item purchased in the transaction.
