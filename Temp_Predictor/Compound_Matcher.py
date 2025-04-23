"""
Author: Suhas Adiga
Affiliation: Theoretical Science Unit (TSU), JNCASR
Date: April 23, 2025
Description: This python code is used to match compounds in user input csv file and SuperCon-MTG by sorting their chemical formula, it rounds off the composition to two decimal places to ensure the accuracy in matching. 
"""
# Importing necessary libraries

import numpy as np 
import pandas as pd
import re

# Loading necessary data files
data_1 = pd.read_csv('Material_prediction.csv')
data_2 = pd.read_csv('../SuperCon-MTG/SuperCon_MTG.csv')

# Function to split the composition
def split_elements(compound):
    splitted_elements = re.findall(r'([A-Z][a-z]?)(\d*\.?\d*)', compound)
    return splitted_elements

# Function to normalize the composition and store it as list
def normalized_counts(elements):
    total = 0
    for element, count in elements:
        if count:
            total += float(count)
        else:
            total += 1 
    normalized_elements = []
    for element, count in elements:
        if count:
            normalized_no = float(count) / total 
        else: 
            normalized_no = 1 / total
        normalized_elements.append((element, np.round(normalized_no,2)))
    normalized_elements.sort()
    return normalized_elements

# Empty dictionary to store composition 
composition_dict = {}
for index, compound in enumerate(data_2['Chemical_Formula']):
    norm_comp = tuple(normalized_counts(split_elements(compound)))
    composition_dict[norm_comp] = (index, compound)

# Empty list to store matched data
matched_data = []

if norm_comp in composition_dict:
   	print(" ╔══════════════════════════════════════════════════════════════════════╗")
   	print(" ║░░░░░░░░░░░░░░░░░░░░░░░░░░░░  WARNING !! ░░░░░░░░░░░░░░░░░░░░░░░░     ║")     
   	print(" ╚══════════════════════════════════════════════════════════════════════╝")

for index, row in data_1.iterrows():
    compound_name = row['Chemical_Formula']
    norm_comp = tuple(normalized_counts(split_elements(compound_name)))
    
    if norm_comp in composition_dict:

    	match_index, match_name = composition_dict[norm_comp]
    	corresponding_row = data_2.iloc[match_index]

    	print(f"A composition similar to {compound_name} was found in SuperCon-MTG: \n Material-ID {corresponding_row['Material-ID']}, composition {match_name} and critical temperature of {corresponding_row['Temp_critical']} K.")