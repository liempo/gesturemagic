## Introduction
`Gesture Magic` is a complex program that maps simple keyboard commands using hand gesture (requires a camera).

## Prerequisites
 - Download and install `Python 3.9` 
 - Clone the repository by either calling `git clone https://github.com/liempo/gesturemagic.git` in the terminal or clicking the clone repository button above.
 - Setup a virtual environment using `virtualenv` package (https://uoa-eresearch.github.io/eresearch-cookbook/recipe/2014/11/26/python-virtual-env/).
 - Activate virtual environment and install required dependencies using `pip install -r requirements.txt`

## Dependencies
- mediapipe==0.8.9.1
- PyAutoGUI==0.9.53
- tensorflow==2.7.0

## Folder and File Structure
 - model – Contains tensorflow files to detect hand gestures
 - hands.py – Contains core code (hand detection and shortcuts)
 - requirements.txt – List of dependencies to be installed

## How to Run
 - Activate virtual env `source env/bin/activate`
 - Run the main file `python hands.py`

### About the Developer
This complex program was developed by Alec Gonzales and James Lumawag, two of the great computer science students from Colegio De San Juan De Letran.
