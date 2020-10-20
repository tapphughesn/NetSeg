# NetSeg

NetSeg is a software that I developed as part of my honors thesis project at UNC-Chapel Hill, department of Computer Science. It allows the user to use a Convolutional Neural Network (CNN) to segment subcortical structures of MR Images of brains (at least 12 months of age).

## Usage

To use this software, you must be in an environment with python 3 installed. Additionally, you will need the SimpleITK and TensorFlow 2 python packages.

For GUI, simply input execute the "NetSeg" binary. Then you will select a T1 image and a T2 image. Then choose your output type (segmentation and/or probability map), and click "Compute Segmentation". If you chose both output types, you will need to specify save paths for the segmentation file and the probability map file, in that order.

For no GUI, simply use the python script "segment.py". Execute "python3 segment.py -h" for details.
