# This code provides analysis about items in Instacart 
# CSV to create association rules between the items 

# Installing "arules" package 
install.packages("arules")
install.packages("tidyverse")

# Loading tidyverse package 
library(tidyverse) 

# Loading arules package 
library(arules) 

# Setting current directory 
setwd("Add File Location Here")

# Loading data from InstacartTransactions.csv 
instacartTransactions <- read.transactions(file = "InstacartTransactions.csv", 
                                           format = "single", 
                                           header = "TRUE", 
                                           sep = ",", 
                                           cols = c("OrderID", "ItemID")) 

# Displaying the summary of tibble 
print(summary(instacartTransactions)) 

# Displaying the first 3 transactions 
inspect(instacartTransactions[1:3])

# Finding frequency of item "24852" 
itemFrequency(instacartTransactions[,"24852"]) 

# Generating Frequency tibble 
instacartTransactionsFrequency <- 
  tibble(Items = names(itemFrequency(instacartTransactions)), 
         Frequency = itemFrequency(instacartTransactions)) 

# Displaying item frequencies 
print(instacartTransactionsFrequency)

# Displaying top 10 frequent purchases 
instacartTransactionsFrequency %>% 
  arrange(desc(Frequency)) %>% 
  slice(1:10) 

# Generating association rules model 
instacartTransactionRules <- apriori(instacartTransactions, 
                                     parameter = list(support = 0.005, 
                                                      confidence = 0.2, 
                                                      minlen = 2)) 

# Displaying summary of model 
summary(instacartTransactionRules) 

# Displaying the top 10 rules 
inspect(instacartTransactionRules[1:10])

# Sorting based on lift and printing top 10 records 
instacartTransactionRules %>% 
  sort(by = "lift") %>% 
  head(n=10) %>% 
  inspect()
