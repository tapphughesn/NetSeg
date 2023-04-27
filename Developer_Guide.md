# NetSeg Developer's Guide

## General Purpose

NetSeg is a program I developed in 2020 alongside my honors thesis. It is intended to be available for use by others as an option for segmenting subcortical structures from MR images. 

## Structure

There are two modules here:

* segment.py: contains code defining a CLI interface and the entire segmentation pipeline. Images are preprocessed 
* model.py: contains code defining the segmentation model (U-Net) that I have used in my project.

segment.py calls model.py to initialize the neural net, which is the most intensive step of the segmentation. You can use the CLI interface in segment.py to perform inference, or you can use a GUI by executing the binary NetSeg, which was compiled for Linux with an AMD CPU. The GUI program was developed in C++ using QT and simply makes a system call to the CLI interface after selection of arguments.

Additionally, there are model weights in the parameters folder. I recommend that these be used for segmentations. There is example data in the templates folder that can be used to visualize a basic segmentation.

## NetSeg Development Process

Some parts of the development of NetSeg are not included here as this is intended to be a minimal, functional, public-facing program. For instance, the training process that was used to generate the model parameters in the parameters folder is not included here. Also, the C++ code used to create the GUI is not here. 

NetSeg was trained on the Longleaf research computing cluster at UNC, and similar training loops to the one used here can be found on the internet. The QT graphical user interface also follows a basic template, using QTCreator, and many similar templates can be found online.

## How to Modify/Expand upon NetSeg

The easiest way to improve NetSeg might be to modify the preprocessing steps in segment.py. There are a few steps which prepare data for segmentation by the neural net, as described in my thesis. However, it is possible that I missed some helpful preprocessing steps that could improve the result! Or, maybe some of these steps aren't needed--you could try omitting them to see how the result is affected.

It is also possible to use my network architechture as a baseline for developing your own 3D segmentation net, or to use my parameters as a starting point for transfer learning training on your own dataset. You can compare the code in model.py to how the architechture is described in my thesis to learn how concept translates to code. In particular, my code here is much simpler than the complex pipelines you will find in many big AI codebases--this code can serve as a good starting point for a student learning tensorflow. 
