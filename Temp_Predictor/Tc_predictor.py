"""
Author: Suhas Adiga
Affiliation: Theoretical Science Unit (TSU), JNCASR
Date: March 11, 2025
Description: This python code is used to predict critical temperature of compounds in the inpur csv file using 30 and 5 key descriptors respectively.
Disclaimer!! 
This code has been converted to PEP-8 format using autopep8.
"""

# Importing necessary libraries

import pandas as pd
import joblib
import seaborn as sns 
import matplotlib.pyplot as plt 

# Loading the CSV File
data_predict = pd.read_csv('../Descriptor_Generator/Material_prediction_desc.csv')

# Loading the trained ML Models 

# Classification model 
model_classify = joblib.load('../Machine_Learning_Models/RF_Classifier.joblib')

# Regression model with 30 descriptors
model_regress = joblib.load('../Machine_Learning_Models/RF_Regressor.joblib')

# Regression model with 5 descriptors 
model_regress_5 = joblib.load('../Machine_Learning_Models/RF_Regressor_5KF.joblib')

# Replace any NaN with 0 (if any)
data_predict.fillna(0)

# DataFrame with 30 descriptors
data_predict_p = data_predict[['Average electronegativity', 'Median electronegativity','Maxima electronegativity', 'Minima electronegativity', 'Range electronegativity', 'Standard deviation electronegativity', 'Average deviation electronegativity', 'Average orbital radius', 'Median orbital radius', 'Maxima orbital radius', 'Minima orbital radius', 'Range orbital radius', 'Standard deviation orbital radius', 'Average deviation orbital radius', 'Median unpaired electron number', 'Maxima unpaired electron number', 'Minima unpaired electron number', 'Range unpaired electron number', 'Standard deviation unpaired electron number', 'Average deviation unpaired electron number', 'Median valence electron number', 'Maxima valence electron number', 'Minima valence electron number', 'Range valence electron number', 'Standard deviation valence electron number', 'Average deviation valence electron number', 'Weighted_avg_en_diff', 'Weighted_avg_orbital_radius_diff', 'Weighted_avg_unpaired_e_no', 'Weighted_avg_valence_no']]

# Prints the shape of the input dataset
print('\n')
print('The size of the dataset is:', data_predict_p.shape[0])

# Classification of compounds to Superconductor and Non-Superconductors
data_predict['Predicted_Superconductor'] = model_classify.predict(data_predict_p)
data_predict['Predicted_Superconductor'] = data_predict['Predicted_Superconductor'].apply(lambda x: 'Superconductor' if x ==1 else 'Non Superconductor')
# Printing classification results
print(data_predict['Predicted_Superconductor'].value_counts())
print("\n╔══════════════════════════════════════════════════╗")
print("║       CRITICAL TEMPERATURE PREDICTION MODELS     ║")
print("╚══════════════════════════════════════════════════╝")

# Dataframe of only predicted superconductors
data_predict_sc = data_predict[data_predict['Predicted_Superconductor'] == 'Superconductor'].copy()

data_predict_nsc = data_predict[data_predict['Predicted_Superconductor'] == 'Non Superconductor'].copy()

# DataFrame with 30 descriptors for predicted superconductors
print("\n Model 1: Full Feature Set")
print("   • Descriptors: 30")
print("   • Task: Predict Tc")

data_predict_sc_r = data_predict_sc[['Average electronegativity', 'Median electronegativity','Maxima electronegativity', 'Minima electronegativity', 'Range electronegativity', 'Standard deviation electronegativity', 'Average deviation electronegativity', 'Average orbital radius', 'Median orbital radius', 'Maxima orbital radius', 'Minima orbital radius', 'Range orbital radius', 'Standard deviation orbital radius', 'Average deviation orbital radius', 'Median unpaired electron number', 'Maxima unpaired electron number', 'Minima unpaired electron number', 'Range unpaired electron number', 'Standard deviation unpaired electron number', 'Average deviation unpaired electron number', 'Median valence electron number', 'Maxima valence electron number', 'Minima valence electron number', 'Range valence electron number', 'Standard deviation valence electron number', 'Average deviation valence electron number', 'Weighted_avg_en_diff', 'Weighted_avg_orbital_radius_diff', 'Weighted_avg_unpaired_e_no', 'Weighted_avg_valence_no']]


# Regression model to predict critical temperature of predicted superconductors using 30 descriptors
data_predict_sc['Predicted_Temp_critical_30_Features'] = model_regress.predict(data_predict_sc_r)

data_predict_nsc['Predicted_Temp_critical_30_Features'] = 0

print("\n Model 2: Reduced Feature Set")
print("   • Descriptors: 5")
print("   • Task: Predict Tc")

# Dataframe of 5 key-features identified using SHAP
data_predict_sc_r_5 = data_predict_sc[['Maxima orbital radius', 'Median electronegativity','Median unpaired electron number', 'Weighted_avg_unpaired_e_no','Median orbital radius' ]]

data_predict_nsc_r_5 = data_predict_nsc[['Maxima orbital radius', 'Median electronegativity','Median unpaired electron number', 'Weighted_avg_unpaired_e_no','Median orbital radius' ]]

# Regression model to predict critical temperature of predicted superconductors using 5 key descriptors
data_predict_sc['Predicted_Temp_critical_5_Features'] = model_regress_5.predict(data_predict_sc_r_5)

data_predict_nsc['Predicted_Temp_critical_5_Features'] = 0 

# Exporting results to csv
data_reduced_sc = pd.DataFrame(data_predict_sc[['Material-ID', 'Chemical_Formula', 'Temp_critical', 'Predicted_Superconductor',
       'Predicted_Temp_critical_30_Features','Predicted_Temp_critical_5_Features' ]])
data_reduced_nsc = pd.DataFrame(data_predict_nsc[['Material-ID', 'Chemical_Formula', 'Temp_critical', 'Predicted_Superconductor',
       'Predicted_Temp_critical_30_Features','Predicted_Temp_critical_5_Features' ]])
data_final = pd.concat([data_reduced_sc, data_reduced_nsc], axis = 0)

# Data Analytics

data_final.to_csv('Material_prediction_results.csv')


#sns.histplot(data_predict_sc['Predicted_Temp_critical_30_Features'])
#plt.title('Histogram of predicted $T_{c}$ with 30 features')
#plt.savefig('Data_Analytics_Tc.png', dpi = 700)