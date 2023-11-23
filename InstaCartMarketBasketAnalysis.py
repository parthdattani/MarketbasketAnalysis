# Install required packages
!pip install mlxtend pandas

# Import necessary libraries
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# Set the current directory
# Replace 'Add File Location Here' with the actual file location
file_location = "Add File Location Here"

# Load data from InstacartTransactions.csv
instacart_df = pd.read_csv(f"{file_location}/InstacartTransactions.csv")

# Displaying the first 3 transactions
print(instacart_df.head(3))

# Convert data to transaction format
te = TransactionEncoder()
transactions = te.fit(instacart_df.groupby('OrderID')['ItemID'].apply(list)).transform(instacart_df['ItemID'].apply(list))

# Displaying the summary of the transactions
print(transactions.shape)

# Displaying item frequencies
item_frequencies = transactions.sum(axis=0)
instacart_transactions_frequency = pd.DataFrame({'Items': te.columns_, 'Frequency': item_frequencies}).sort_values(by='Frequency', ascending=False)

# Displaying top 10 frequent purchases
print(instacart_transactions_frequency.head(10))

# Generating association rules model
frequent_itemsets = apriori(transactions, min_support=0.005, use_colnames=True)
instacart_transaction_rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.2)

# Displaying summary of the model
print(instacart_transaction_rules.info())

# Displaying the top 10 rules
print(instacart_transaction_rules.head(10))

# Sorting based on lift and printing top 10 records
instacart_transaction_rules_lift = instacart_transaction_rules.sort_values(by="lift", ascending=False).head(10)
print(instacart_transaction_rules_lift)
