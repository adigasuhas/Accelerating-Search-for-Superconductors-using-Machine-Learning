# Accelerating Search for Superconductors using Machine Learning

<div align="center">
    <img src="/Other_Files/Comp_SC_1.png" width="250">
</div>


[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/release/python-310/)
[![Paper](https://img.shields.io/badge/paper-arXiv-blue)](https://arxiv.org/abs/your-paper-id)
![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)  
![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)  

## Description

This repository contains code used for my project: *Accelerating Search for Superconductors using Machine Learning*. 

## Requirements

- **Python 3.10+** is required. You can check your Python version using:

  ```bash
  python3 --version

## Usage

- Clone this repository to your local machine:

  ```bash
  git clone https://github.com/adigasuhas/Accelerating-Search-for-Superconductors-using-Machine-Learning.git
  ```

- Download the trained models from Google Drive: [Click Here](https://drive.google.com/drive/folders/18K2BYny9yomTUyySdaZeJ_XK9qTRr3u4?usp=drive_link)

- Ensure to rename the unzipped folder as `Machine_Learning_Models`.

- Move the `Machine_Learning_Models` folder to the cloned repository:

  ```bash
  mv Machine_Learning_Models Accelerating-Search-for-Superconductors-using-Machine-Learning/
  ```

- Navigate to the code directory where the Tc prediction scripts are located:

  ```bash
  cd Accelerating-Search-for-Superconductors-using-Machine-Learning/Temp_Predictor/
  ```
## Preparing the Input File

- Open the file named `Material_prediction.csv`.
- The following columns can be handled as described:

    - `Material-ID` (_Optional_): This is for user reference and identification. It is not auto-filled by the code, so you may enter any identifier (e.g., Unique Identification number, Sample name).
    - `Chemical_Formula`: Add the materials (chemical compositions) for which you wish to predict the critical temperature ($T_c$).
    - `Temp_critical`(_Optional_): If known, you may enter the experimental $T_c$ here for comparison. Otherwise, you may leave it blank.
      
- Follow the formatting rules illustrated in the reference image provided in the repository to ensure your chemical composition input is valid.

    <div align="center">
        <img src="/Other_Files/Chemical_Composition_Rules.png" width="850">
    </div>

## Compound Similarity Check

- After entering the `Chemical_Formula` into `Material_prediction.csv`, you can check if the compound is present in the `SuperCon-MTG` dataset by running the following command. A warning will be displayed, and if the compound is found, its corresponding material ID and critical temperature will be shown.

    ```bash
    cd Temp_Predictor
    python3 Compound_Matcher.py
    ```
    <div align="center">
        <img src="/Other_Files/Comp_match_warning.png" width="450">
    </div>


## Generating Descriptors and Critical Temperature Prediction

- Once you have validated that the compound is not present in the `SuperCon-MTG` dataset, you can proceed to generate descriptors for $T_c$ prediction. To do this, simply run the `Predict.sh` script, which will generate the descriptors and predict $T_c$ using both the 30-feature and 5-feature models. 

    If the script does not have execute permissions, you can grant them using the following command (skip if already granted):

    ```bash
    chmod +700 Predict.sh
    ./Predict.sh
    ```

- Upon successful execution, a `Material_prediction_results.csv` file will be generated inside the `Temp_Predictor` folder. This file will contain the following columns:
  - `Material-ID`
  - `Chemical_Formula`
  - `Temp_critical`
  - `Predicted_class`
  - `Predicted_Temp_critical_30_Features`
  - `Predicted_Temp_critical_5_Features`

## References

This work has been made possible due to the insights and datasets provided by the following sources, which have served as key inspirations:

1. **Original Dataset**:
   - Center for Basic Research on Materials. *MDR SuperCon Datasheet Ver.240322*. National Institute for Materials Science, 2024. [DOI: 10.48505/NIMS.4487](https://mdr.nims.go.jp/pid/650c4826-f0ca-42e8-8dd9-94025a5307ce)

2. **Descriptors**:
   - Rabe, K. M., Phillips, J. C., Villars, P., & Brown, I. D. (1992). *Global multinary structural chemistry of stable quasicrystals, high-$(T_c)$ ferroelectrics, and high-$(T_c)$ superconductors*. Phys. Rev. B, 45(14), 7650–7676. [DOI: 10.1103/PhysRevB.45.7650](https://link.aps.org/doi/10.1103/PhysRevB.45.7650)

3. **Composition Generation**:
   - Davies, D., Butler, K., Jackson, A., Skelton, J., Morita, K., & Walsh, A. (2019). *SMACT: Semiconducting Materials by Analogy and Chemical Theory*. Journal of Open Source Software, 4(38), 1361. [DOI: 10.21105/joss.01361](http://dx.doi.org/10.21105/joss.01361)

4. **Machine Learning-Based Work**:
   - Stanev, V., Oses, C., Kusne, A. G., Rodriguez, E., Paglione, J., Curtarolo, S., & Takeuchi, I. (2018). *Machine learning modeling of superconducting critical temperature*. npj Computational Materials, 4(1). [DOI: 10.1038/s41524-018-0085-8](http://dx.doi.org/10.1038/s41524-018-0085-8)

## Our Work

```
@article{Adiga2025-dt,
  title  = "Accelerating the Search for Superconductors Using Machine Learning",
  author = "Adiga, S and Waghmare, U V",
  year   =  2025
}

```
