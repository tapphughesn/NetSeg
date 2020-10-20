import tensorflow as tf
import os
import SimpleITK as sitk
import sys
import argparse
import numpy as np
from datetime import datetime
from model import *
from glob import glob

# dont hardcode temp file names or template
# use process id in temp file name
# use configure.py to store variables like template, cropping size, probability threshold, etc


# Main Function
def main(main_args):
    
    t1_image = main_args.t1
    t2_image = main_args.t2
    working_dir = main_args.work_dir
    seg_save_path = main_args.seg_path
    prob_map_save_path = main_args.prob_map_path

    print("working directory: " + working_dir)

    if ((t1_image == "") and (t2_image == "")):
        sys.exit("T1 and T2 images were not provided")
    elif (t1_image == ""):
        sys.exit("T1_image was not provided")
    elif (t2_image == ""):
        sys.exit("T2_image was not provided")

    #use SimpleITK to read images

    reader = sitk.ImageFileReader()
    reader.SetFileName(t1_image)
    t1_image = reader.Execute()
    reader.SetFileName(t2_image)
    t2_image = reader.Execute()
    reader.SetFileName(working_dir + "/../NetSeg/templates/1year-Average-IBIS-MNI-t1w-stripped.nrrd")
    t1_reference_image = reader.Execute()
    reader.SetFileName(working_dir + "/../NetSeg/templates/1year-Average-IBIS-MNI-t2w-stripped.nrrd")
    t2_reference_image = reader.Execute()

    # do histomatching

    matcher = sitk.HistogramMatchingImageFilter()
    matcher.ThresholdAtMeanIntensityOn()
    matcher.SetNumberOfHistogramLevels(1024)
    matcher.SetNumberOfMatchPoints(7)
    t1_image = matcher.Execute(t1_image, t1_reference_image)
    t2_image = matcher.Execute(t2_image, t2_reference_image)

    # Get arrays

    t1_array = sitk.GetArrayFromImage(t1_image)
    t2_array = sitk.GetArrayFromImage(t2_image)
    t1_array = np.asarray(t1_array)
    t2_array = np.asarray(t2_array)

    # Account for the differences between pynrrd loading and sitk loading... :(

    t1_array = np.swapaxes(t1_array,0,2)
    t2_array = np.swapaxes(t2_array,0,2)

    # do cropping and 99th percentile normalizing

    # cropping to the central rectangular prism

    (dim1, dim2, dim3) = np.shape(t1_array)
    (buf1, buf2, buf3) = tuple(np.subtract(np.shape(t1_array), (96,112,96))//2)
    t1_array = t1_array[(buf1+1):(dim1-buf1),(buf2+1):(dim2-buf2),(buf3+1):(dim3-buf3)]

    t2_array = t2_array[(buf1+1):(dim1-buf1),(buf2+1):(dim2-buf2),(buf3+1):(dim3-buf3)]

    # 99th percentile normalizing
    k = int(np.floor(np.product(np.shape(t1_array))*0.99))
    almost_max = np.partition(t1_array, k, axis=None)[k]
    t1_array = t1_array/almost_max

    k = int(np.floor(np.product(np.shape(t2_array))*0.99))
    almost_max = np.partition(t2_array, k, axis=None)[k]
    t2_array = t2_array/almost_max

    # convert t1 and t2 images to tensorflow tensors and concatenate them along last axis

    t1_array = tf.convert_to_tensor(t1_array, dtype = tf.float32)
    t2_array = tf.convert_to_tensor(t2_array, dtype = tf.float32)
    t1_array = tf.expand_dims(t1_array, 3)
    t2_array = tf.expand_dims(t2_array, 3)
    input_array = tf.concat([t1_array, t2_array], axis=3)
    input_array = tf.expand_dims(input_array, 0)
    
    # Get model
    model = unet_4()
    model_checkpoint_load_path = working_dir + "/../NetSeg/parameters/parameters_10-11-2020-06:58:26"
    model.load_weights(model_checkpoint_load_path)

    # write probability maps as 3D images for each class -- nvm use sitk.compose()
    
    prob_map_array = np.asarray(model(input_array, training_bool = False))[0]
    # seg_array = np.argmax(prob_map_array, axis=3)
    seg_array = np.argmax(prob_map_array, axis=3)*(np.max(prob_map_array, axis=3) > 0.5)

    # POST-PROCESSING STEP: get rid of stray voxels

    for j in range(len(seg_array)):
        for k in range(len(seg_array[j])):
            for l in range(len(seg_array[j,k])):

                # borders are background
                if(j == 0 or j == len(seg_array)-1 or k == 0 or k == len(seg_array[0])-1 or l == 0 or l == len(seg_array[0,0])-1):
                    seg_array[j,k,l] = 0
                    continue

                adj = np.zeros(7)
                # Now need to check categories of all 26 adjacent voxels
                # (a,b,c) will be the offset of the indices j,k,l to access a adjacent point
                for n in range(1,27):
                    a = (n // 9) - 1
                    b = (n % 9 // 3) - 1
                    c = (n % 9 % 3) - 1
                    adj[seg_array[j+a,k+b,l+c]] += 1

                # if all but three neighboring voxels are the same label
                # change this voxel's label to that
                if (np.max(adj) >= 22):
                    seg_array[j,k,l] = np.argmax(adj)

    # create threshold for maximum probability so that if the network does not give a class probability over
    # the threshold, the voxel is marked as "uncertain"

    # uncrop segmentation by adding 0 padding

    cropped_seg_array = seg_array
    padding = np.zeros((dim1, dim2, dim3))
    padding[(buf1+1):(dim1-buf1),(buf2+1):(dim2-buf2),(buf3+1):(dim3-buf3)] = seg_array
    seg_array = padding

    # uncrop probmap array
    zeros = np.zeros((dim1, dim2, dim3, 7))
    zeros[(buf1+1):(dim1-buf1),(buf2+1):(dim2-buf2),(buf3+1):(dim3-buf3),:] = prob_map_array
    prob_map_array = zeros

    # Swap axes back before saving! :(

    seg_array = np.swapaxes(seg_array,0,2)
    prob_map_array = np.swapaxes(prob_map_array,0,2)

    writer = sitk.ImageFileWriter()

    if (seg_save_path != ""):
        seg_img = sitk.GetImageFromArray(seg_array)
        seg_img.CopyInformation(t1_image)
        s_filter = sitk.CastImageFilter()
        s_filter.SetOutputPixelType(sitk.sitkUInt16)
        seg_img = s_filter.Execute(seg_img)
        writer.SetFileName(seg_save_path)
        writer.Execute(seg_img)

    if (prob_map_save_path != ""):
        prob_map_img = sitk.GetImageFromArray(prob_map_array)
        writer.SetFileName(prob_map_save_path)
        writer.Execute(prob_map_img)

if (__name__ == "__main__"):
    parser = argparse.ArgumentParser(description="Segments brain image")
    parser.add_argument("--t1", help="Specify T1 modality image", default="")
    parser.add_argument("--t2", help="Specify T2 modality image", default="")
    parser.add_argument("--work_dir", help="The path of NetRun executable", default="")
    parser.add_argument("--seg_path", help="Specify save path for output segmentation", default="")
    parser.add_argument("--prob_map_path", help="Specify save path for output probability map", default = "")
    main(parser.parse_args())
    
