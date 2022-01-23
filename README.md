## Introduction
`Gesture Magic` is a complex program that maps simple keyboard commands using hand gesture (requires a camera).

## Prerequisites
 - Download and install `Python 3.9` 
 - Clone the repository by either calling `git clone https://github.com/liempo/gesturemagic.git` in the terminal or clicking the clone repository button above.
 - Setup a virtual environment using `virtualenv` package (https://uoa-eresearch.github.io/eresearch-cookbook/recipe/2014/11/26/python-virtual-env/).
 - Activate virtual environment and install required dependencies using `pip install -r requirements.txt`

## Dependencies
- absl-py==1.0.0
- astunparse==1.6.3
- attrs==21.2.0
- cachetools==4.2.4
- certifi==2021.10.8
- charset-normalizer==2.0.9
- cycler==0.11.0
- flatbuffers==2.0
- fonttools==4.28.4
- gast==0.4.0
- google-auth==2.3.3
- google-auth-oauthlib==0.4.6
- google-pasta==0.2.0
- grpcio==1.43.0
- h5py==3.6.0
- idna==3.3
- importlib-metadata==4.9.0
- keras==2.7.0
- Keras-Preprocessing==1.1.2
- kiwisolver==1.3.2
- libclang==12.0.0
- Markdown==3.3.6
- matplotlib==3.5.1
- mediapipe==0.8.9.1
- MouseInfo==0.1.3
- numpy==1.21.4
- oauthlib==3.1.1
- opencv-contrib-python==4.5.4.60
- opencv-python==4.5.4.60
- opt-einsum==3.3.0
- packaging==21.3
- Pillow==8.4.0
- protobuf==3.19.1
- pyasn1==0.4.8
- pyasn1-modules==0.2.8
- PyAutoGUI==0.9.53
- PyGetWindow==0.0.9
- PyMsgBox==1.0.9
- pyparsing==3.0.6
- pyperclip==1.8.2
- PyRect==0.1.4
- PyScreeze==0.1.28
- python-dateutil==2.8.2
- pytweening==1.0.4
- requests==2.26.0
- requests-oauthlib==1.3.0
- rsa==4.8
- six==1.16.0
- tensorboard==2.7.0
- tensorboard-data-server==0.6.1
- tensorboard-plugin-wit==1.8.0
- tensorflow==2.7.0
- tensorflow-estimator==2.7.0
- tensorflow-io-gcs-filesystem==0.23.1
- termcolor==1.1.0
- typing_extensions==4.0.1
- urllib3==1.26.7
- Werkzeug==2.0.2
- wrapt==1.13.3
- zipp==3.6.0

## Folder and File Structure
 - model – Contains tensorflow files to detect hand gestures
 - hands.py – Contains core code (hand detection and shortcuts)
 - requirements.txt – List of dependencies to be installed

## How to Run
 - Activate virtual env `source env/bin/activate`
 - Run the main file `python hands.py`

### About the Developer
This complex program was developed by Alec Gonzales and James Lumawag, two of the great computer science students from Colegio De San Juan De Letran.
