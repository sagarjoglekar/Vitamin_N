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

def getDownloadedImages(ImageDir):
    try:
        images =['_'.join(k.split('/')[-1].split('_')[:2]) for k in glob.glob(ImageDir +'*.jpg')]
        return images
    except:
        return []
    
    

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
    cityDir = '../Data/GreaterLondon/'
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
        downloadedImages = getDownloadedImages(cityDir + 'GSV_Downloads/')
        filteredPanoMap = {k:panoMap[k] for k in panoMap.keys() if k not in downloadedImages} 
        keyidx = i%len(keylist)
        print('Using Key: %s',keylist[keyidx])
        downloadImages(cityDir,'GSV_Downloads/',filteredPanoMap,120,0,0,keylist[keyidx])
        
        