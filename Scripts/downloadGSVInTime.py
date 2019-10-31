import urllib,urllib2
import xmltodict
import cStringIO
import ogr, osr
import time
import os,os.path
import PanoDownload 
import json
import glob
import os
import requests
import shutil
import itertools
from PIL import Image
from io import BytesIO

def getDownloadedKeys(ImageDir):
    images = glob.glob(ImageDir +'*.jpg')
    

def downloadImages(cityDir, imageDir, imageMapping, fov, pitch, heading, key):
    width = 400
    height = 400
    downloadDir = cityDir + imageDir
    if not os.path.exists(downloadDir):
        os.makedirs(downloadDir)
    url = "https://maps.googleapis.com/maps/api/streetview"
    
    for k in imageMapping:
        for pano in imageMapping[k]:
            filename = k + '_' + pano
            panoid = imageMapping[k][pano]
            print("Downloading:", panoid)
            params = {
                # maximum permitted size for free calls
                "size": "%dx%d" % (width, height),
                "fov": fov,
                "pitch": pitch,
                "heading": heading,
                "pano": panoid,
                "key": key
            }
            response = requests.get(url, params=params, stream=True)
            try:
                img = Image.open(BytesIO(response.content))
                filename = '%s/%s.%s' % (downloadDir, filename, 'jpg')
                img.save(filename)
            except:
                print("Image not found")
                filename = None
            del response
        # let the code to pause by 1s, in order to not go over data limitation of Google quota
        time.sleep(1)
    
            
            
            
def extractMappings(file):
    PointDict = {}
    record = json.load(open(file , 'rb'))
    for k in record:
        PointDict[k] = {}
        for pano in record[k]:
            if 'year' in pano:
                nameFrag = str(pano['year']) + '_' + str(pano['month'])
                PointDict[k][nameFrag] = pano['panoid']             
    return PointDict
        
        
if __name__ == '__main__':
    cityDir = '../Data/Cambridge/'
    key_file = 'keys.txt'
    files= glob.glob(cityDir + '*.json')    
    
    lines = open(key_file,"r")
    keylist = []
    for line in lines:
        key = line[:-1]
        keylist.append(key)
    
    print ('The key list is:=============', keylist)
    for i in range(len(files)):
        panoMap = extractMappings(files[i]) 
        keyidx = i%len(keylist)
        downloadImages(cityDir,'GSV_Downloads/',panoMap,120,0,0,keylist[keyidx])
        
        