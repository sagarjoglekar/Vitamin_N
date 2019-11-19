import numpy as np
import os.path
import argparse
import math
import cv2
import sys
import time
import glob
import csv
import collections

# Make sure that caffe is on the python path:
caffe_root = '/datasets_1/sagarj/caffe-segnet-cudnn5/'
sys.path.insert(0, caffe_root + 'python')
import caffe

caffe.set_mode_gpu()

modelDef = "/work/sagarj/Work/BellLabs/Segnet-Tutorial/Example_Models/segnet_model_driving_webdemo.prototxt"
modelWeights = "/work/sagarj/Work/BellLabs/Segnet-Tutorial/trained_weights/segnet_weights_driving_webdemo.caffemodel"
color_file = "/work/sagarj/Work/BellLabs/Segnet-Tutorial/Scripts/camvid11.png"

parser = argparse.ArgumentParser()


parser.add_argument('--dataDir', type=str, required=True)
parser.add_argument('--outDir' , type=str , required=True)
parser.add_argument('--outFile' , type=str , required=False)


args = parser.parse_args()

dataDir = args.dataDir
outDir = args.outDir
outFile = args.outFile

if outFile == None:
        outFile = 'segnetLabels.csv'
        print("Saving labels to: %s",outFile)
else:
        print("Saving labels to: %s",outFile)

Image_List = glob.glob(dataDir+"/*.jpg")
print("Found %d images for SegNet",len(Image_List))


labels = ['Sky', 'Building', 'Pole','Road_Marking','Road','Pavement','Tree','Sign_Symbol','Fence','Vehicle','Pedestrian', 'Bike']

print Image_List[1]


def normalizeDict(d, denom):
    normDict = {}
    for k in d : 
        normDict[k] = float(d[k])/float(denom)
    return normDict

net = caffe.Net(modelDef,modelWeights,caffe.TEST)

input_shape = net.blobs['data'].data.shape
output_shape = net.blobs['argmax'].data.shape
label_colours = cv2.imread(color_file).astype('uint8')

print(input_shape,output_shape)
fieldNames = ['Imgkey'] + labels


with open(outFile,'wb') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldNames)
        writer.writeheader()

        for file_path in Image_List:
                record = {}
                start = time.time()
                key = file_path.split('/')[-1].split('.')[0]
                record['Imgkey'] = key
                input_image_raw = caffe.io.load_image(file_path)
                input_image = caffe.io.resize_image(input_image_raw, (input_shape[2],input_shape[3]))
                input_image = input_image*255
                input_image = input_image.transpose((2,0,1))
                input_image = input_image[(2,1,0),:,:]
                input_image = np.asarray([input_image])
                input_image = np.repeat(input_image,input_shape[0],axis=0)
                net.blobs['data'].data[...] = input_image
                out = net.forward()
                end = time.time()
                print('Executed SegNet in: ', str((end - start)*1000), 'ms')
                start = time.time()
                segmentation_ind = net.blobs['argmax'].data.copy()
                segLabels = np.squeeze(segmentation_ind)
                denom = segLabels.shape[0]*segLabels.shape[1]
                ratios =  normalizeDict(collections.Counter(np.squeeze(segLabels.flatten())),denom)
                
                for i in range(len(labels)):
                        if i in ratios:
                                record[labels[i]] = ratios[i]
                        else:
                                record[labels[i]] = 0.0
                writer.writerow(record)
