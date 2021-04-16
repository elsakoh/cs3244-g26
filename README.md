# Fall-Net

Classify videos of human activity as fall or not-fall events. CS3244 AY20/21 Sem 2 Project.

This repository uses `python 3.x` and is tested on `python 3.6`

## Datasets
Model has been trained on these public datasets: [URFD](http://fenix.univ.rzeszow.pl/~mkepski/ds/uf.html) and
[MCFD](http://www.iro.umontreal.ca/~labimage/Dataset/).

Training and testing datasets used
- [SISFD](http://sistemic.udea.edu.co/en/investigacion/proyectos/english-falls/)
- [NTU-RGBD](http://rose1.ntu.edu.sg/datasets/actionrecognition.asp), only the FALL and ADL (Activities of Daily living) datasets

## Project structure
### Preprocessing 
Video and Image Preprocessing files are found in `dataset_preprocessing/`. 
preprocessing is split into two parts: 
1. formatting videos into frames `preprocess_XX.py`
2. generating optical flow images `generate_OF_XX.py`

The preprocessed images are used as inputs to train the model.

### Model
Refer to file `temporalnet.py`. Each model works specifically on each of the fall datasets.

Refer to paper cited below for the model's architecture.

### Running the model
1. Install dependencies `pip3 install -r requirements.txt`
2. `python temporalnet.py`

## References:

[Fall-Detection-with-CNNs-and-Optical-Flow](https://github.com/AdrianNunez/Fall-Detection-with-CNNs-and-Optical-Flow)

This repository contains code from the paper:
```
Núñez-Marcos, A., Azkune, G., & Arganda-Carreras, I. (2017).
"Vision-Based Fall Detection with Convolutional Neural Networks"
Wireless Communications and Mobile Computing, 2017.
```

## Contributors: 
Abhinav Ramnath , Carel Chay Jia Ming, Koh Hui Hui Elizabeth
Fang Pin Sern, Goh Jia Jun, Jordan Rahul Sabanayagam