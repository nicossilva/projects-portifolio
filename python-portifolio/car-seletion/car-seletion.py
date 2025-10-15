# Code based on matrix weighted rank for decision-making

#importations
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Criteria weight factor (Wi)
# Define the criteria weight based on your perception
if os.path.exists("criteria.csv"): #check if the file exist, else show error msg
    df_classification = pd.read_csv("criteria.csv") #import csv file to the variable "df"
else:
    print("File not found!")

#import data files
if os.path.exists("car-list.csv"): #check if the file exist, else show error msg
    car_df = pd.read_csv("car-list.csv") #import csv file to the variable "df"
else:
    print("File not found!")

#data manipulation
car_df["weight_hp"] = car_df["weight"]/car_df["hp"] #create a new column of weight/hp ratio
qualitative_map = {"Good":1.0,"Ok":0.7,"Bad":0.5} #transform qualitative measure into numerical analysis
car_df["consumption"] = car_df["consumption"].replace(",",".",regex = True).astype(float)
car_df["aceleration"] = car_df["aceleration"].replace(",",".",regex = True).astype(float)

#transform qualitative info into quantitative info
car_df["confort_num"] = car_df["confort"].map(qualitative_map) #assign numbers to confort level column in a number
car_df["design_num"] = car_df["design"].map(qualitative_map) #assign number to the design column

#Definition of numeric values for the qualitative data
def classify(value): #define de condition to apply the transformation
    if value == "electric": #evaluate if the value is electric or not
        return 1.0
    elif value == "hydraulic" : #evalute if the value is electric or not
        return 0.5
    else:
        return "info not found!"

car_df["steering_num"] = car_df["steering"].apply(classify) #create a new column to transform the qualitative data into quantitative

## Normalization of the data
columns_norm = ["price","insurance","hp","aceleration","consumption","warranty", "weight_hp"] #select the columns for the normalization
car_df[columns_norm] = MinMaxScaler().fit_transform(car_df[columns_norm]) #normalization of the data in the columns
car_df[["price","insurance","consumption"]] = 1-car_df[["price","insurance","consumption"]] #invert_columns = ["price","insurance","consumption"]

#Calculation of the weighted value
columns_calc = ["price","insurance","hp", "weight_hp","aceleration","consumption","steering_num","design_num","warranty","confort_num"] #select the columns that will be used for the analysis

car_df[columns_calc] = car_df[columns_calc].fillna(0) #change blank spaces for zero for the multiplication
df_weightedvalue = car_df[columns_calc].mul(df_classification.iloc[0]) #multiply the value of each car for the weight of importance
car_df["final_point"] = df_weightedvalue.sum(axis=1) #Final sum for each car

#Print
car_df = car_df.sort_values("final_point", ascending = False) #sort the cars by the pontuation on descending mode
print(car_df[["Car-name","final_point"]])
