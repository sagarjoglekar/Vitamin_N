## To crawl mappiliary images you will need to use the following notebooks in sequence

### 1. CityAcquisition_images.ipynb
This notebook would allow you to grab street network for a selected city and then sample points in an equidistant fashion along the road networks.
The output of this process is another shapefile with the locations of these points

### 2. Mapilliary_Meta_download.ipynb
Not all the points sampled in step 1 will have mappiliary images captured. For that reason, we make calls to the mappilary api for every selected point, 
to return to us IDs of images in the X meters vicinity of the sampled point from step 1. 
The output of this process are a series of json files which contain the information of different assets to be found on mappiliary, with their metadata and locations

### 3. Mapilliary_Image_download.ipynb
In this final step, we would collate all the locations and assets found in step 2 from mappiliary, into one coherent dataframe. We will then download these 
images from the mappiliary map api using the IDs collated from step 2. 