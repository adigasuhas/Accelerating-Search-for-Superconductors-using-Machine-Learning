"""
Author: Suhas Adiga
Affiliation: Theoretical Sciences Unit (TSU), JNCASR
Date: February 21, 2025 (Re-verified)
Description: This python code generates descriptors and features used for our machine learning model to predict critical temperature of superconductors.
# !! Disclaimer !!
# This code is a mix of five different codes. It works, but it’s not the best or most efficient. Someday, it should be cleaned up with loops or functions—just not today. 
# This code was converted to PEP 8 guidelines using autopep8.
"""

# Importing necessary libraries 

import pandas as pd 
import re 
import numpy as np
from qsd_params import valency, electronegativity, orbital_radius, unpaired_electron_no

# Loading the 'SuperCon-MTG' database along or user can provided CSV filesfor descriptor generation outside the database. 

data = pd.read_csv('../Temp_Predictor/Material_prediction.csv')

print(
    "\033[1;36m"
    "╔════════════════════════════════════════════════════════╗\n"
    "║                                                        ║\n"
    "║   \033[1;33mA C C E L E R A T I N G   S E A R C H   F O R\033[1;36m        ║\n"
    "║   \033[1;33mS U P E R C O N D U C T O R S   U S I N G\033[1;36m            ║\n"
    "║   \033[1;33mM A C H I N E   L E A R N I N G\033[1;36m                      ║\n"
    "║                                                        ║\n"
    "╚════════════════════════════════════════════════════════╝\n"
    " \033[1;32mBy Suhas Adiga, Ram Seshadri, and Umesh Waghmare\033[0m\n"
)


# Creating list of metals and non-metals 
# Note: Metalloids are considered as non-metals

non_metals = ['H','D','T','He','B','C','N','O','F','Ne','Si','P','S','Cl','Ar','Ge','As','Se','Br','Kr','Sb','Te','I','Xe','At','Rn','Ts','Og']

metals = ['Li','Be','Na','Mg','Al','K','Ca','Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn','Ga',
          'Rb','Sr','Y','Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd','In','Sn','Cs','Ba','La','Ce',
          'Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu','Hf','Ta','W','Re','Os','Ir',
          'Pt','Au','Hg','Tl','Pb','Bi','Po','Fr','Ra','Ac','Th','Pa','U','Np','Pu','Am','Cm','Bk','Cf','Es',
          'Fm','Md','No','Lr','Rf','Db','Sg','Bh','Hs','Mt','Ds','Rg','Cn','Nh','Fl','Mc','Lv']

# Function to split a string of chemical compounds into a list of elements and their stoichiometry.

def split_elements_with_composition(compound):
	splitted_elements = re.findall(r'([A-Z][a-z]?)(\d*\.?\d*)', compound)
	return splitted_elements

# Function to split a string of chemical compounds into a list of elements without their stoichiometry.

def split_elements_without_composition(compound):
    splitted_elements = re.findall(r'([A-Z][a-z]?)(\d*\.?\d*)', compound)
    return [el[0] for el in splitted_elements]

# Function to normalize the composition 

def normalized_counts(elements):
    # Calculating total composition 
    total = 0 # set initial sum of composition as 0
    for element, count in elements:
        if count:
            total += float(count) 
        else:
            total += 1 # If stoichiometry is not mentioned, it is 1. 
    # Empty list to store element and its normalized composition
    normalized_elements = [] 
    for element, count in elements:
        if count:
            normalized_no = float(count)/total
        else:
            normalized_no = 1/total
        normalized_elements.append((element, normalized_no))
    return normalized_elements

# Function to normalize only metallic elements stoichiometry in the composition
# E.g. Na2Mg2Cl3, for this compound on normalization, Na -> 0.5 Mg -> 0.5 is taken 

def normalized_counts_metal(elements):
    """Function to normalize metallic composition in a chemical formula"""
    total = 0
    for element, count in elements:
        if element not in non_metals:
            if count:
                total += float(count)
            else:
                total += 1
    normalized_elements_metal = []
    for element , count in elements:
        if element not in non_metals:
            if count:
                normalised_no = float(count)/total 
            else: 
                normalised_no = 1/total
            normalized_elements_metal.append((element,normalised_no))
        if element not in metals:
            if count:
                normalised_no = 0
            else:
                normalised_no =0
            normalized_elements_metal.append((element,normalised_no))
    return normalized_elements_metal    

# Function to get composition from the database 

def get_composition(row):
    composition = {}
    # The element columns start from the 4th column. 
    for element in row.index[3:]:
        if pd.notna(row[element]) and row[element] > 0:
            composition[element] = row[element]
    return composition 

# Normalizing the composition in the input data file 

#for compound in data['Chemical_Formula']:
#    print(normalized_counts(split_elements_with_composition(compound)))    

# index_2 dynamically creates columns with element names as headers.  
for index_1, compound in enumerate(data['Chemical_Formula']):
    normalized_elements_c = normalized_counts(split_elements_with_composition(compound))
    for index_2, count in normalized_elements_c:
        data.at[index_1,index_2] = float(count)

# Fills missing entries with 0 for better readability.  

normalized_data_all = data.fillna(0)

# Normalizing the metallic composition and store it as data_metal 

data_metal = data.loc[:, ['Material-ID', 'Chemical_Formula', 'Temp_critical']]

for index_1, compound in enumerate(data_metal['Chemical_Formula']):
    normalized_elements_m = normalized_counts_metal(split_elements_with_composition(compound))
    for element, count in normalized_elements_m:  # Renaming index_2 to 'element' for clarity
        if element not in data_metal.columns:  # Add new column if missing
            data_metal[element] = 0.0
        data_metal.at[index_1, element] = float(count)



# normalized_data_all.to_csv('Predict_compounds_norm.csv')

# Generate Statistical Features 
# Function to calculate statistical features related to electronegativity 

def calculate_electronegativity_features(composition,feature):
    weighted_array = np.array([composition[element]*electronegativity(element) for element in composition]) # Features are calculated by using the normalized stoichiometry as the weights. 

    if len(weighted_array) == 0:
        return np.nan 

    if feature == 'average':
        return sum(weighted_array)/sum(composition.values()) if sum(composition.values()) != 0 else np.nan
    elif feature == 'median':
        return np.median(weighted_array)
    elif feature == 'variance':
        return np.var(weighted_array)
    elif feature == 'maxima':
        return np.max(weighted_array)
    elif feature == 'minima':
        return np.min(weighted_array)
    elif feature == 'range':
        return np.max(weighted_array) - np.min(weighted_array)
    elif feature == 'std_dev':
        return np.std(weighted_array)
    elif feature == 'avg_dev':
        return np.mean(np.abs(weighted_array - np.mean(weighted_array)))

# Function to calculate statistical features related to orbital radius  

def calculate_orbital_radius_features(composition,feature):
    weighted_array = np.array([composition[element]*orbital_radius(element) for element in composition]) # Features are calculated by using the normalized stoichiometry as the weights. 

    if len(weighted_array) == 0:
        return np.nan 

    if feature == 'average':
        return sum(weighted_array)/sum(composition.values()) if sum(composition.values()) != 0 else np.nan
    elif feature == 'median':
        return np.median(weighted_array)
    elif feature == 'variance':
        return np.var(weighted_array)
    elif feature == 'maxima':
        return np.max(weighted_array)
    elif feature == 'minima':
        return np.min(weighted_array)
    elif feature == 'range':
        return np.max(weighted_array) - np.min(weighted_array)
    elif feature == 'std_dev':
        return np.std(weighted_array)
    elif feature == 'avg_dev':
        return np.mean(np.abs(weighted_array - np.mean(weighted_array)))

#Note: Average is not calculated for unpaired electron number and valence electron number as they correspond to the weighted average of valence electron number (from QSD) and unpaired electron number.

# Function to calculate statistical features related to unpaired electron number  

def calculate_unpaired_e_no_features(composition,feature):
    weighted_array = np.array([composition[element]*unpaired_electron_no(element) for element in composition]) # Features are calculated by using the normalized stoichiometry as the weights. 

    if len(weighted_array) == 0:
        return np.nan 

    if feature == 'median':
        return np.median(weighted_array)
    elif feature == 'variance':
        return np.var(weighted_array)
    elif feature == 'maxima':
        return np.max(weighted_array)
    elif feature == 'minima':
        return np.min(weighted_array)
    elif feature == 'range':
        return np.max(weighted_array) - np.min(weighted_array)
    elif feature == 'std_dev':
        return np.std(weighted_array)
    elif feature == 'avg_dev':
        return np.mean(np.abs(weighted_array - np.mean(weighted_array)))

# Function to calculate statistical features related to valence electron number  

def calculate_valence_e_no_features(composition,feature):
    weighted_array = np.array([composition[element]*valency(element) for element in composition]) # Features are calculated by using the normalized stoichiometry as the weights. 

    if len(weighted_array) == 0:
        return np.nan 

    if feature == 'median':
        return np.median(weighted_array)
    elif feature == 'variance':
        return np.var(weighted_array)
    elif feature == 'maxima':
        return np.max(weighted_array)
    elif feature == 'minima':
        return np.min(weighted_array)
    elif feature == 'range':
        return np.max(weighted_array) - np.min(weighted_array)
    elif feature == 'std_dev':
        return np.std(weighted_array)
    elif feature == 'avg_dev':
        return np.mean(np.abs(weighted_array - np.mean(weighted_array)))

# Function to calculated Weighted average of metallic electronegativity difference

def weighted_average_metallic_electronegativity_difference(row):
    composition = {}

    # Collect non-zero element compositions
    for element in row.index[3:]:
        if pd.notna(row[element]) and row[element] > 0:
            composition[element] = row[element]

    elements = list(composition.keys())
    proportions = list(composition.values())

    # Get the order of elements from the compound name
    compound = row['Chemical_Formula']
    compound_elements = split_elements_without_composition(compound)
    
    # Sort elements and proportions by alphabetical order in compound name when proportions are equal else in the increasing order of proportions as described in Quantum Structure Diagrams. 
    sorted_elements = [x for _, x in sorted(zip(proportions, elements), key=lambda pair: (pair[0], compound_elements.index(pair[1])))]
    sorted_proportions = sorted(proportions)

    # Initialize the weighted average of metallic electronegativity radius difference
    weighted_diff = 0

    # Calculate the weighted average of metallic electronegativity difference based on the number of elements
    num_elements = len(sorted_elements)
    for i in range(num_elements):
        for j in range(i + 1, num_elements):
            # The weight is twice the least of the compositions of the two elements
            weight = 2 * min(sorted_proportions[i], sorted_proportions[j])
            radius_diff = electronegativity(sorted_elements[i]) - electronegativity(sorted_elements[j])
            weighted_diff += weight * radius_diff

    return weighted_diff

# Function to calculated Weighted average of orbital radius difference 
def weighted_average_orbital_radius_difference(row):
    composition = {}

    # Collect non-zero element compositions
    for element in row.index[3:]:
        if pd.notna(row[element]) and row[element] > 0:
            composition[element] = row[element]

    elements = list(composition.keys())
    proportions = list(composition.values())

    # Get the order of elements from the compound name
    compound = row['Chemical_Formula']
    compound_elements = split_elements_without_composition(compound)
    
    # Sort elements and proportions by alphabetical order in compound name when proportions are equal else in the increasing order of proportions as described in Quantum Structure Diagrams.
    sorted_elements = [x for _, x in sorted(zip(proportions, elements), key=lambda pair: (pair[0], compound_elements.index(pair[1])))]
    sorted_proportions = sorted(proportions)

    # Initialize the weighted average orbital radius difference
    weighted_diff = 0

    # Calculate the weighted average orbital radius difference based on the number of elements
    num_elements = len(sorted_elements)
    for i in range(num_elements):
        for j in range(i + 1, num_elements):
            # The weight is twice the least of the compositions of the two elements
            weight = 2 * min(sorted_proportions[i], sorted_proportions[j])
            radius_diff = orbital_radius(sorted_elements[i]) - orbital_radius(sorted_elements[j])
            weighted_diff += weight * radius_diff

    return weighted_diff

# Function to calculate Weighted average of unpaired electron number 

def weighted_average_unpaired_e_no(row):
    composition = {}

    for element in row.index[3:]:
        if pd.notna(row[element]) and row[element] > 0:
            composition[element] = row[element]

    weighted_sum = sum(composition[element] * unpaired_electron_no(element) for element in composition)
    total_composition = sum(composition.values())

    if total_composition == 0:
        return 0
    else:
        return weighted_sum / total_composition

# Function to calculate Weighted average of valence electron number 
def weighted_average_valence_e_no(row):
    composition = {}

    for element in row.index[3:]:
        if pd.notna(row[element]) and row[element] > 0:
            composition[element] = row[element]

    weighted_sum = sum(composition[element] * valency(element) for element in composition)
    total_composition = sum(composition.values())

    if total_composition == 0:
        return 0
    else:
        return weighted_sum / total_composition

# Creating an empty dataframe for Statistical Features
stat_features = pd.DataFrame(index=data.index)
compositions = data.apply(get_composition, axis=1)

# Creating an empty DataFrame for QSD Features
qsd_features = pd.DataFrame(index = data.index)

# Adding each statistical feature as a column 
# a] Electronegativity related features 
print("╔══════════════════════════════════════════════════╗")
print("║           FEATURE GENERATION SUMMARY             ║")
print("╚══════════════════════════════════════════════════╝")

stat_features['Average electronegativity'] = compositions.apply(lambda comp: calculate_electronegativity_features(comp, 'average'))
stat_features['Median electronegativity'] = compositions.apply(lambda comp: calculate_electronegativity_features(comp, 'median'))
stat_features['Maxima electronegativity'] = compositions.apply(lambda comp:calculate_electronegativity_features(comp, 'maxima'))
stat_features['Minima electronegativity'] = compositions.apply(lambda comp: calculate_electronegativity_features(comp, 'minima'))
stat_features['Range electronegativity'] = compositions.apply(lambda comp:calculate_electronegativity_features(comp, 'range'))
stat_features['Standard deviation electronegativity'] = compositions.apply(lambda comp: calculate_electronegativity_features(comp, 'std_dev'))
#stat_features['Variance electronegativity'] = compositions.apply(lambda comp:calculate_electronegativity_features(comp, 'variance'))
stat_features['Average deviation electronegativity'] = compositions.apply(lambda comp: calculate_electronegativity_features(comp, 'avg_dev'))

print("\n🔹Statistical features related to Electronegativity           ✓ Generated")

# b] Orbital Radius related features 

stat_features['Average orbital radius'] = compositions.apply(lambda comp: calculate_orbital_radius_features(comp, 'average'))
stat_features['Median orbital radius'] = compositions.apply(lambda comp: calculate_orbital_radius_features(comp, 'median'))
stat_features['Maxima orbital radius'] = compositions.apply(lambda comp:calculate_orbital_radius_features(comp, 'maxima'))
stat_features['Minima orbital radius'] = compositions.apply(lambda comp: calculate_orbital_radius_features(comp, 'minima'))
stat_features['Range orbital radius'] = compositions.apply(lambda comp:calculate_orbital_radius_features(comp, 'range'))
stat_features['Standard deviation orbital radius'] = compositions.apply(lambda comp: calculate_orbital_radius_features(comp, 'std_dev'))
#stat_features['Variance orbital radius'] = compositions.apply(lambda comp:calculate_orbital_radius_features(comp, 'variance'))
stat_features['Average deviation orbital radius'] = compositions.apply(lambda comp: calculate_orbital_radius_features(comp, 'avg_dev'))

print("\n🔹Statistical features related to Orbital Radius              ✓ Generated")

# c] Unpaired electron number  

stat_features['Median unpaired electron number'] = compositions.apply(lambda comp: calculate_unpaired_e_no_features(comp, 'median'))
stat_features['Maxima unpaired electron number'] = compositions.apply(lambda comp:calculate_unpaired_e_no_features(comp, 'maxima'))
stat_features['Minima unpaired electron number'] = compositions.apply(lambda comp: calculate_unpaired_e_no_features(comp, 'minima'))
stat_features['Range unpaired electron number'] = compositions.apply(lambda comp:calculate_unpaired_e_no_features(comp, 'range'))
stat_features['Standard deviation unpaired electron number'] = compositions.apply(lambda comp: calculate_unpaired_e_no_features(comp, 'std_dev'))
#stat_features['Variance unpaired electron number'] = compositions.apply(lambda comp:calculate_unpaired_e_no_features(comp, 'variance'))
stat_features['Average deviation unpaired electron number'] = compositions.apply(lambda comp: calculate_unpaired_e_no_features(comp, 'avg_dev'))

print("\n🔹Statistical features related to Unpaired electron number    ✓ Generated")

# d] Valence electron number  

stat_features['Median valence electron number'] = compositions.apply(lambda comp: calculate_valence_e_no_features(comp, 'median'))
stat_features['Maxima valence electron number'] = compositions.apply(lambda comp:calculate_valence_e_no_features(comp, 'maxima'))
stat_features['Minima valence electron number'] = compositions.apply(lambda comp: calculate_valence_e_no_features(comp, 'minima'))
stat_features['Range valence electron number'] = compositions.apply(lambda comp:calculate_valence_e_no_features(comp, 'range'))
stat_features['Standard deviation valence electron number'] = compositions.apply(lambda comp: calculate_valence_e_no_features(comp, 'std_dev'))
#stat_features['Variance unpaired electron number'] = compositions.apply(lambda comp:calculate_valence_e_no_features(comp, 'variance'))
stat_features['Average deviation valence electron number'] = compositions.apply(lambda comp: calculate_valence_e_no_features(comp, 'avg_dev'))

print("\n🔹Statistical features related to Valence electron number     ✓ Generated")


# Adds weighted average of metalic electronegativity difference as a column in the output
qsd_features['Weighted_avg_en_diff'] = data_metal.apply(weighted_average_metallic_electronegativity_difference, axis = 1)

# Adds weighted average of orbital radius difference as a column in the output
qsd_features['Weighted_avg_orbital_radius_diff'] = data.apply(weighted_average_orbital_radius_difference, axis =1)

# Adds weighted average of unpaired electron number as a column in the output
qsd_features['Weighted_avg_unpaired_e_no'] = data.apply(weighted_average_unpaired_e_no, axis = 1)

# Adds weighted average of valence electron number as a column in the output
qsd_features['Weighted_avg_valence_no'] = data.apply(weighted_average_valence_e_no, axis = 1)

print("\n🔹Quantum Structural Diagram based descriptors                ✓ Generated")

# Removes columns that contain element-wise composition data 
data_reduced = data[['Material-ID', 'Chemical_Formula', 'Temp_critical']]


# Exporting the generated features and descriptors as CSV
features = pd.concat([data_reduced, stat_features, qsd_features], axis=1)

features.to_csv('Material_prediction_desc.csv', index=False)