# CMSC491
This is course project for CMSC491 related to domain adaptation semantic segmentation

# Instruction for runing code
To run this you need to first setup the docker 
Then you need to download the checkpoints if you want to see the result without any training 
after that you need to download the full dataset, sample dataset is provided here
finally open the Code_run.ipnyb file and run the code

## Docker installation
1. Open the Makefile
2. In line 11 change the workspace accordingly
3. Open ubuntu terminal in the docker_env directory
4. TYPE: 
		$ make docker-build
		$ make docker-run
5. Open any web browser
6. On the bookmarks bar type: 
		$ localhost:17888/
7. You will see the whole docker environment

## Run Code
To run the code you need to download two things

1. Checkpoints: Links are provided in the Checkpoints folder
2. Dataset: Sample dataset is provided here, but to train the model you need to download full dataset

### Dataset link

1. Cityscape: https://www.cityscapes-dataset.com/
2. Synthia: https://www.v7labs.com/open-datasets/synthia-dataset

### Training

1. First open the train.py and change all the argument accordingly
2. Open the Code_run.ipynb and run the Train Phase

### Testing

1. Open the Code_run.ipynb
2. Follow the steps 

> Sample output is provided in the Code_run.ipynb file 