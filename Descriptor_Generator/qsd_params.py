"""
Author: Suhas Adiga
Affiliation: Theoretical Science Unit (TSU), JNCASR
Date: June 11, 2024
Description: This python code contains the values of valence electrons, unpaired electron number, electronegativity in Martynov-Batsanov Scale and Zunger pseudo potential cutoff radius.

Note: We have interpreted the Zunger Pseudopotential cutoff radius for the element Curium as [R(Am)*R(Gd)]/R(Eu)

"""
# Importing necessary libaries
import math

# Dictionary with values of valence electrons in ground / stable
# electronic state

valence_electron = {
    'H': 1, 'D': 1, 'T': 1, 'He': 2,
    'Li': 1, 'Be': 2, 'B': 3, 'C': 4, 'N': 5, 'O': 6, 'F': 7, 'Ne': 8,
    'Na': 1, 'Mg': 2, 'Al': 3, 'Si': 4, 'P': 5, 'S': 6, 'Cl': 7, 'Ar': 8,
    'K': 1, 'Ca': 2, 'Sc': 2, 'Ti': 2, 'V': 2, 'Cr': 1, 'Mn': 2, 'Fe': 2, 'Co': 2, 'Ni': 2, 'Cu': 1, 'Zn': 2, 'Ga': 3, 'Ge': 4, 'As': 5, 'Se': 6, 'Br': 7, 'Kr': 8,
    'Rb': 1, 'Sr': 2, 'Y': 2, 'Zr': 2, 'Nb': 1, 'Mo': 1, 'Tc': 2, 'Ru': 1, 'Rh': 1, 'Pd': 10, 'Ag': 1, 'Cd': 2, 'In': 3, 'Sn': 4, 'Sb': 5, 'Te': 6, 'I': 7, 'Xe': 8,
    'Cs': 1, 'Ba': 2, 'La': 2, 'Ce': 2, 'Pr': 2, 'Nd': 2, 'Pm': 2, 'Sm': 2, 'Eu': 2, 'Gd': 2, 'Tb': 2, 'Dy': 2, 'Ho': 2, 'Er': 2, 'Tm': 2, 'Yb': 2, 'Lu': 2, 'Hf': 2, 'Ta': 2, 'W': 2, 'Re': 2, 'Os': 2, 'Ir': 2, 'Pt': 1, 'Au': 1, 'Hg': 2, 'Tl': 3, 'Pb': 4, 'Bi': 5, 'Po': 6, 'At': 7, 'Rn': 8,
    'Fr': 1, 'Ra': 2, 'Ac': 2, 'Th': 2, 'Pa': 2, 'U': 2, 'Np': 2, 'Pu': 2, 'Am': 2, 'Cm': 2, 'Bk': 2, 'Cf': 2, 'Es': 2, 'Fm': 2, 'Md': 2, 'No': 2, 'Lr': 3, 'Rf': 2, 'Db': 2, 'Sg': 2, 'Bh': 2, 'Hs': 2, 'Mt': 2, 'Ds': 2, 'Rg': 2, 'Cn': 2, 'Nh': 3, 'Fl': 4, 'Mc': 5, 'Lv': 6, 'Ts': 7, 'Og': 8

}

# Dictionary with values of electronegativity of elements in Martynov -
# Batsanov scale

electronegativity_mbscale = {
    'H': 2.20, 'D': 2.20, 'T': 2.20, 'He': math.nan,
    'Li': 0.9, 'Be': 1.45, 'B': 1.90, 'C': 2.37, 'N': 2.85, 'O': 3.32, 'F': 3.78, 'Ne': math.nan,
    'Na': 0.89, 'Mg': 1.31, 'Al': 1.64, 'Si': 1.98, 'P': 2.32, 'S': 2.65, 'Cl': 2.98, 'Ar': math.nan,
    'K': 0.80, 'Ca': 1.17, 'Sc': 1.50, 'Ti': 1.86, 'V': 2.22, 'Cr': 2.00, 'Mn': 2.04, 'Fe': 1.67, 'Co': 1.72, 'Ni': 1.76, 'Cu': 1.08, 'Zn': 1.44, 'Ga': 1.70, 'Ge': 1.99, 'As': 2.27, 'Se': 2.54, 'Br': 2.83, 'Kr': 3.00,
    'Rb': 0.80, 'Sr': 1.13, 'Y': 1.41, 'Zr': 1.70, 'Nb': 2.03, 'Mo': 1.94, 'Tc': 2.18, 'Ru': 1.97, 'Rh': 1.99, 'Pd': 2.08, 'Ag': 1.07, 'Cd': 1.40, 'In': 1.63, 'Sn': 1.88, 'Sb': 2.14, 'Te': 2.38, 'I': 2.76, 'Xe': 2.60,
    'Cs': 0.77, 'Ba': 1.08, 'La': 1.35, 'Ce': 1.12, 'Pr': 1.13, 'Nd': 1.14, 'Pm': 1.15, 'Sm': 1.17, 'Eu': 1.15, 'Gd': 1.2, 'Tb': 1.1, 'Dy': 1.22, 'Ho': 1.23, 'Er': 1.24, 'Tm': 1.25, 'Yb': 1.1, 'Lu': 1.27, 'Hf': 1.73, 'Ta': 1.94, 'W': 1.79, 'Re': 2.06, 'Os': 1.85, 'Ir': 1.87, 'Pt': 1.91, 'Au': 1.19, 'Hg': 1.49, 'Tl': 1.69, 'Pb': 1.92, 'Bi': 2.14, 'Po': 2.40, 'At': 2.64, 'Rn': 2.2,
    'Fr': 0.79, 'Ra': 0.9, 'Ac': 1.1, 'Th': 1.3, 'Pa': 1.5, 'U': 1.38, 'Np': 1.36, 'Pu': 1.28, 'Am': 1.13, 'Cm': 1.28, 'Bk': 1.3, 'Cf': 1.3, 'Es': 1.3, 'Fm': 1.3, 'Md': 1.3, 'No': 1.3, 'Lr': 1.3, 'Rf': math.nan, 'Db': math.nan, 'Sg': math.nan, 'Bh': math.nan, 'Hs': math.nan, 'Mt': math.nan, 'Ds': math.nan, 'Rg': math.nan, 'Cn': math.nan, 'Nh': math.nan, 'Fl': math.nan, 'Mc': math.nan, 'Lv': math.nan, 'Ts': math.nan, 'Og': math.nan

}

# Dictionary with values of Zunger Pseudopotential cutoff radius of elements.

zungerpp_radius = {
    'H': 1.25, 'D': 1.25, 'T': 1.25, 'He': math.nan,
    'Li': 1.61, 'Be': 1.08, 'B': 0.795, 'C': 0.64, 'N': 0.54, 'O': 0.465, 'F': 0.405, 'Ne': math.nan,
    'Na': 2.65, 'Mg': 2.03, 'Al': 1.675, 'Si': 1.42, 'P': 1.24, 'S': 1.10, 'Cl': 1.01, 'Ar': math.nan,
    'K': 3.69, 'Ca': 3.00, 'Sc': 2.75, 'Ti': 2.58, 'V': 2.43, 'Cr': 2.44, 'Mn': 2.22, 'Fe': 2.11, 'Co': 2.02, 'Ni': 2.18, 'Cu': 2.04, 'Zn': 1.88, 'Ga': 1.695, 'Ge': 1.56, 'As': 1.415, 'Se': 1.285, 'Br': 1.20, 'Kr': math.nan,
    'Rb': 4.10, 'Sr': 3.21, 'Y': 2.94, 'Zr': 2.825, 'Nb': 2.76, 'Mo': 2.72, 'Tc': 2.65, 'Ru': 2.605, 'Rh': 2.52, 'Pd': 2.45, 'Ag': 2.375, 'Cd': 2.215, 'In': 2.05, 'Sn': 1.88, 'Sb': 1.765, 'Te': 1.67, 'I': 1.585, 'Xe': math.nan,
    'Cs': 4.31, 'Ba': 3.402, 'La': 3.08, 'Ce': 4.50, 'Pr': 4.48, 'Nd': 3.99, 'Pm': 3.99, 'Sm': 4.14, 'Eu': 3.94, 'Gd': 3.91, 'Tb': 3.89, 'Dy': 3.67, 'Ho': 3.65, 'Er': 3.63, 'Tm': 3.60, 'Yb': 3.59, 'Lu': 3.37, 'Hf': 2.91, 'Ta': 2.79, 'W': 2.735, 'Re': 2.68, 'Os': 2.65, 'Ir': 2.628, 'Pt': 2.70, 'Au': 2.66, 'Hg': 2.41, 'Tl': 2.235, 'Pb': 2.09, 'Bi': 1.997, 'Po': 1.90, 'At': 1.83, 'Rn': math.nan,
    'Fr': 4.37, 'Ra': 3.53, 'Ac': 3.12, 'Th': 4.98, 'Pa': 4.96, 'U': 4.72, 'Np': 4.93, 'Pu': 4.91, 'Am': 4.89, 'Cm': 4.85, 'Bk': math.nan, 'Cf': math.nan, 'Es': math.nan, 'Fm': math.nan, 'Md': math.nan, 'No': math.nan, 'Lr': math.nan, 'Rf': math.nan, 'Db': math.nan, 'Sg': math.nan, 'Bh': math.nan, 'Hs': math.nan, 'Mt': math.nan, 'Ds': math.nan, 'Rg': math.nan, 'Cn': math.nan, 'Nh': math.nan, 'Fl': math.nan, 'Mc': math.nan, 'Lv': math.nan, 'Ts': math.nan, 'Og': math.nan

}

# Dictionary with values of unpaired electron number of elements

unpaired_electron = {
    'H': 1, 'D': 1, 'T': 1, 'He': 0,
    'Li': 1, 'Be': 0, 'B': 1, 'C': 2, 'N': 3, 'O': 2, 'F': 1, 'Ne': 0,
    'Na': 1, 'Mg': 0, 'Al': 1, 'Si': 2, 'P': 3, 'S': 2, 'Cl': 1, 'Ar': 0,
    'K': 1, 'Ca': 0, 'Sc': 1, 'Ti': 2, 'V': 3, 'Cr': 4, 'Mn': 5, 'Fe': 4, 'Co': 3, 'Ni': 2, 'Cu': 1, 'Zn': 0, 'Ga': 1, 'Ge': 2, 'As': 3, 'Se': 2, 'Br': 1, 'Kr': 0,
    'Rb': 1, 'Sr': 0, 'Y': 1, 'Zr': 2, 'Nb': 3, 'Mo': 4, 'Tc': 5, 'Ru': 4, 'Rh': 3, 'Pd': 2, 'Ag': 1, 'Cd': 0, 'In': 1, 'Sn': 2, 'Sb': 3, 'Te': 2, 'I': 1, 'Xe': 0,
    'Cs': 1, 'Ba': 0, 'La': 1, 'Ce': 2, 'Pr': 3, 'Nd': 4, 'Pm': 5, 'Sm': 6, 'Eu': 7, 'Gd': 8, 'Tb': 9, 'Dy': 4, 'Ho': 3, 'Er': 2, 'Tm': 1, 'Yb': 0, 'Lu': 1, 'Hf': 2, 'Ta': 3, 'W': 4, 'Re': 5, 'Os': 4, 'Ir': 3, 'Pt': 2, 'Au': 1, 'Hg': 0, 'Tl': 1, 'Pb': 2, 'Bi': 3, 'Po': 2, 'At': 1, 'Rn': 0,
    'Fr': 1, 'Ra': 0, 'Ac': 1, 'Th': 2, 'Pa': 3, 'U': 4, 'Np': 5, 'Pu': 6, 'Am': 7, 'Cm': 8, 'Bk': 9, 'Cf': 4, 'Es': 3, 'Fm': 2, 'Md': 1, 'No': 0, 'Lr': 1, 'Rf': 2, 'Db': 3, 'Sg': 4, 'Bh': 5, 'Hs': 4, 'Mt': 3, 'Ds': 2, 'Rg': 1, 'Cn': 0, 'Nh': 1, 'Fl': 2, 'Mc': 3, 'Lv': 2, 'Ts': 1, 'Og': 0
}


# Function to fetch electronegativity of element
def electronegativity(element):
    return electronegativity_mbscale[element]

# Function to fetch valency of element
def valency(element):
    return valence_electron[element]

# Function to fetch orbital radius of element
def orbital_radius(element):
    return zungerpp_radius[element]

# Function to fetch unpaired electron number of element
def unpaired_electron_no(element):
    return unpaired_electron[element]
