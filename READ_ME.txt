Instructions for running main.py:

This script was originally create in PyCharm. 
Install Keras, Tensorflow or Tensorflow-GPU, matplotlib, and numpy to run the script. If you have a graphics card, I strongly recommend installing Tensorflow-GPU as it will greatly speed up training time. 

Because of size limitations on Github, unzip the three dataset folders and copy those three folders into a new folder. Call that new folder ir_dataset. Main.py is setup to look for this folder. 

A train model is includes. Have ir_ident_model.h5 in the same folder as the dataset and main.py for the script to recognize and use the file.

If you wish to train a new model from scratch using the dataset of thermal images I have created, change the continue_training variable at the top of the script to False and remove the .h5 model file from the folder. 

The dataset it quite small, only a little more than two-thousand images. After many iterations of the model, I found the dataset it too small to make predictions that are better than about 60% accurate. 