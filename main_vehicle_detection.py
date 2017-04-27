#import the required modules and functions
import cv2
from os import listdir,chdir
from matplotlib import pyplot as plt
import numpy as np
import pickle
from moviepy.editor import VideoFileClip
from skimage.feature import hog
import matplotlib.image as mpimg
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import svm
import time
import random
from sklearn.utils import shuffle
from scipy.ndimage.measurements import label
from collections import deque
from required_functions import *

#set the working directory
chdir("C:\\Users\\che\\CarND-Vehicle-Detection\\")

#list of folders with vehicle images
path = ["C:\\Users\\che\\CarND-Vehicle-Detection\\vehicles\\GTI_Far\\",
        "C:\\Users\\che\\CarND-Vehicle-Detection\\vehicles\GTI_Left\\",
        "C:\\Users\\che\\CarND-Vehicle-Detection\\vehicles\\GTI_MiddleClose\\",
        "C:\\Users\\che\\CarND-Vehicle-Detection\\vehicles\\GTI_Right\\",
        "C:\\Users\\che\\CarND-Vehicle-Detection\\vehicles\\KITTI_extracted\\"]

#list of folders with non-vehicle images
notcars_path = ["C:\\Users\\che\\CarND-Vehicle-Detection\\non-vehicles\\GTI\\",
                "C:\\Users\\che\\CarND-Vehicle-Detection\\non-vehicles\\Extras\\"]


#create empty lists for appending cars, notcars, path of car and notcars images
cars = []
notcars = []
cars_path = []
not_cars_path=[]


#Read the car images and append to the car list.
#Append the path of car images to the cars_path list.
for i in range(5):
    filename = listdir(path[i])
    for names in filename:
        image = cv2.imread(path[i]+names)
        cars.append(image)
        cars_path.append(path[i]+names)

#Read the noncar images and append to the notcars list
#Append the path of noncars to the not_cars_path
for i in range(2):
    filename_notcars = listdir(notcars_path[i])
    for names in filename_notcars:
        image = cv2.imread(notcars_path[i]+names)
        notcars.append(image)
        not_cars_path.append(notcars_path[i]+names)

#Check the lengh of lists
print(len(cars))
print(len(notcars))
print(len(cars_path))
print(len(not_cars_path))


#Arguments for extracting the features from car and noncar images
color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 8 # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
spatial_size = 24 # Spatial binning dimensions
hist_bins = 8    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
y_start_stop = [None, None] # Min and max in y to search in slide_window()


#Extract the car and noncar image features
car_features = extract_features(cars_path, color_space=color_space, spatial_size=(spatial_size,spatial_size),
                        hist_bins=hist_bins, orient=orient, 
                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel,
                        spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)

not_car_features = extract_features(not_cars_path,color_space=color_space, spatial_size=(spatial_size,spatial_size),
                        hist_bins=hist_bins, orient=orient, 
                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel,
                        spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)



if len(car_features) > 0:
    # Create an array stack of  car and noncar features
    X = np.vstack((car_features, not_car_features)).astype(np.float64)                        
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)
    car_ind = np.random.randint(0, len(cars))
    # Plot an example of raw and scaled features
    fig = plt.figure(figsize=(12,4))
    plt.subplot(131)
    plt.imshow(cars[car_ind])
    plt.title('Original Image')
    plt.subplot(132)
    plt.plot(X[car_ind])
    plt.title('Raw Features')
    plt.subplot(133)
    plt.plot(scaled_X[car_ind])
    plt.title('Normalized Features')
    fig.tight_layout()
else: 
    print('Your function only returns empty feature vectors...')
    

# Create labels for car and noncar features
y = np.hstack((np.ones(len(car_features)), np.zeros(len(not_car_features))))


#Split up data into randomized training and test sets
#set seed 
seed=7
#split the data into test and train sets randomly
X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=0.2, random_state=seed)
print('Using:',orient,'orientations',pix_per_cell,
    'pixels per cell and', cell_per_block,'cells per block')
print('Feature vector length:', len(X_train[0]))
# Use a linear SVC 
svc = svm.LinearSVC()
# Check the training time for the SVC
t=time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')
# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
# Check the prediction time for a single sample
t=time.time()


#Serialize the important variables and save it to file
dist_pickle = {}
dist_pickle["svc"] = svc
dist_pickle["X_scaler"] = X_scaler
dist_pickle["car_features"] = car_features
dist_pickle["not_car_features"] = not_car_features 
dist_pickle["X_train"] = X_test
dist_pickle["X_test"] = X_test
dist_pickle["y_train"] = X_test
dist_pickle["y_test"] = X_test 
with open("saved_pickle.p", "wb") as f:
        pickle.dump(dist_pickle, f)

#load the variables and check the accuracy again 
dist_pickle = pickle.load(open("./saved_pickle.p", "rb" ))
svc = dist_pickle["svc"]
X_scaler = dist_pickle["X_scaler"]
car_features = dist_pickle["car_features"]
not_car_features = dist_pickle["not_car_features"]
X_test = dist_pickle["X_test"]
y_test = dist_pickle["y_test"]
X_train = dist_pickle["X_train"]
y_train = dist_pickle["y_train"]   
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))

global_heat = []
global_heat = deque(maxlen=15)
maxlength = 16
threshold = 1
x_start_stop = [600,1280]
xy_overlap = (0.9,0.9)


def final_pipeline(image):
    heat = np.zeros_like(image[:,:,0]).astype(np.float)
    xy_window = (64,64)
    y_start_stop = [380,440]
    windows = slide_window(image, x_start_stop=x_start_stop, y_start_stop=y_start_stop, 
                   xy_window=xy_window, xy_overlap=xy_overlap)
    hot_windows = search_windows(image, windows, svc, X_scaler, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)
    heat = add_heat(heat,hot_windows)
    
    global global_heat
    
    if len(global_heat) < maxlength:
        global_heat.append(heat)
    else:
        global_heat.popleft()
        global_heat.append(heat)
        
        
    xy_window = (128,128)
    y_start_stop = [400,656]
    windows = slide_window(image, x_start_stop=x_start_stop, y_start_stop=y_start_stop, 
                    xy_window=xy_window, xy_overlap=xy_overlap)
    hot_windows = search_windows(image, windows, svc, X_scaler, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)
    heat = add_heat(heat,hot_windows)
    if len(global_heat) < maxlength:
        global_heat.append(heat)
    else:
        global_heat.popleft()
        global_heat.append(heat)
        
    xy_window = (256,256)
    y_start_stop = [600,656]
    windows = slide_window(image, x_start_stop=x_start_stop, y_start_stop=y_start_stop, 
                    xy_window=xy_window, xy_overlap=xy_overlap)
    hot_windows = search_windows(image, windows, svc, X_scaler, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)
    heat = add_heat(heat,hot_windows)
    if len(global_heat) < maxlength:
        global_heat.append(heat)
    else:
        global_heat.popleft()
        global_heat.append(heat)
   
    
    stacked_heat = np.array(global_heat)
    stacked_heat = np.reshape(stacked_heat,(1,-1,720,1280))
    stacked_heat = np.vstack(stacked_heat)
    average_heat = np.mean((stacked_heat),axis=0)
    heatmap = apply_threshold(average_heat,threshold)
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(np.copy(image),labels)
    return draw_img

#Read the video frame by frame. Apply the pipeline and write the video to file    
output = 'submission.mp4'
clip1 = VideoFileClip("project_video.mp4")
white_clip = clip1.fl_image(final_pipeline) 
%time white_clip.write_videofile(output, audio=False)



'''
Code for generating plots

'''


#Plot a random car and notcar image and save to the folder 
car_ind = np.random.randint(0,len(cars))
notcar_ind = np.random.randint(0,len(notcars))
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(5,5))
ax1.imshow(cars[car_ind])
ax1.set_title('Example_car', fontsize=8)
ax2.imshow(notcars[notcar_ind])
ax2.set_title('Example_Not_car', fontsize=8)
name = 'car_notcar_example'
plt.savefig(name)


# code for plotting HOG visualisation
ind = np.random.randint(0, len(cars))
# Read in the image
image = mpimg.imread(cars[ind])
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
# Define HOG parameters
orient = 9
pix_per_cell = 8
cell_per_block = 2
# Call our function with vis=True to see an image output
features, hog_image = get_hog_features(gray, orient, 
                        pix_per_cell, cell_per_block, 
                        vis=True, feature_vec=False)

car_ind = np.random.randint(0,len(cars_path))
notcar_ind = np.random.randint(0,len(not_cars_path))
car_img = mpimg.imread(cars_path[car_ind])
notcar_img = mpimg.imread(not_cars_path[notcar_ind])
cgray = cv2.cvtColor(car_img,cv2.COLOR_RGB2GRAY)
ngray = cv2.cvtColor(notcar_img,cv2.COLOR_RGB2GRAY)
cfeature,chog_image = get_hog_features(cgray,orient,pix_per_cell,
                                      cell_per_block,vis=True,feature_vec=False)
nfeature,nhog_images = get_hog_features(ngray,orient,pix_per_cell,
                                      cell_per_block,vis=True,feature_vec=False)

fig = plt.figure()
plt.subplot(121)
plt.imshow(car_img)
plt.title('Example Car Image')
plt.subplot(122)
plt.imshow(chog_image, cmap='gray')
plt.title('HOG Visualization')

fig = plt.figure()
plt.subplot(121)
plt.imshow(notcar_img)
plt.title('Example Non Car Image')
plt.subplot(122)
plt.imshow(nhog_images, cmap='gray')
plt.title('HOG Visualization')


#code for plotting spatial bin features
car_ind = np.random.randint(0,len(cars_path))
notcar_ind = np.random.randint(0,len(not_cars_path))
car_img = mpimg.imread(cars_path[car_ind])
notcar_img = mpimg.imread(not_cars_path[notcar_ind])
color_car = cv2.cvtColor(car_img,cv2.COLOR_RGB2YCrCb)
color_non = cv2.cvtColor(notcar_img,cv2.COLOR_RGB2YCrCb)
color_feat = cv2.resize(color_car,(24,24))
noncolor_feat = cv2.resize(color_non,(24,24))

fig = plt.figure()
plt.subplot(121)
plt.imshow(car_img)
plt.title('Example Car Image')
plt.subplot(122)
plt.imshow(color_feat)
plt.title('Color Feature')

fig = plt.figure()
plt.subplot(121)
plt.imshow(notcar_img)
plt.title('Example Non Car Image')
plt.subplot(122)
plt.imshow(noncolor_feat)
plt.title('Color Feature')


#The final pipeline function was slightly modifeid and this output of heatmaps
#were produced
path = "C:\\Users\\che\\CarND-Vehicle-Detection\\screenshots\\"
image_list = listdir(path)
for name in image_list:
    image = cv2.imread(path+name)
    heat = heatmap(image)
    names = 'heat_'+name+'.jpg'
    cv2.imwrite(names,heat)

#The final pipeline function was slightly modifeid and this output of labeled images
#were produced    
path = "C:\\Users\\che\\CarND-Vehicle-Detection\\screenshots\\"
image_list = listdir(path)
for name in image_list:
    image = cv2.imread(path+name)
    heat = label_image(image)
    names = 'label_'+name+'.jpg'
    cv2.imwrite(names,heat)


#code for seeing the final output on test images
img = cv2.imread("C:/Users/che/CarND-Vehicle-Detection/test_images/test1.jpg")
image = final_pipeline(img)
plt.imshow(image)

img = cv2.imread("C:/Users/che/CarND-Vehicle-Detection/test_images/test2.jpg")
image = final_pipeline(img)
plt.imshow(image)

img = cv2.imread("C:/Users/che/CarND-Vehicle-Detection/test_images/test3.jpg")
image = final_pipeline(img)
plt.imshow(image)

img = cv2.imread("C:/Users/che/CarND-Vehicle-Detection/test_images/test4.jpg")
image = final_pipeline(img)
plt.imshow(image)

img = cv2.imread("C:/Users/che/CarND-Vehicle-Detection/test_images/test5.jpg")
image = final_pipeline(img)
plt.imshow(image)

img = cv2.imread("C:/Users/che/CarND-Vehicle-Detection/test_images/test6.jpg")
image = final_pipeline(img)
plt.imshow(image)


