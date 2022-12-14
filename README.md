# Priv_NF   
Adam Taras 2022
Work done under Donald G. Dansereau as part of the Robotic Imaging Group, The University of Sydney

This is a git repo for source code for the second half of my thesis, with an aim of quantifying the privacy of our proposed methods. 

### Requirements

Python 3.9

```
matplotlib==3.5.3
numpy==1.23.2
opencv_python==4.6.0.66
pytorch_lightning==1.7.7
scipy==1.9.0
seaborn==0.12.1
torchvision==0.13.1+cu113
```


### Structure

The folders for this project are:
- NF: (Normalising Flows) contains an adaptation of the [UvA tutes](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial11/NF_image_modeling.html) for training and evaluating the fidelity of flow models. 
- hashes: an implementation of random circle and random line extrema
- optimization_attack: a least squares based optimization attack on the hashes
- tests: unittsts on base functionality
- data_manipulation: misc tools for transforming LSUN bedroom data into a consistent form


### Pre trained models

Pre-trained models can be found [here](https://drive.google.com/file/d/1pfLjzSLIvJj8NA_1KXAn1-7D7-uXx7HV/view?usp=share_link)


