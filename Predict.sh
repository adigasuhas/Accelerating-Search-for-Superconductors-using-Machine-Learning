#!/bin/bash

cd Descriptor_Generator
python3 Master_Data_Generator.py

cd ../Temp_Predictor/
python3 Tc_predictor.py


