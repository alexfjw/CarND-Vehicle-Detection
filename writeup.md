## Writeup Template
### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[regionandscales]: ./output_images/region2.png
[beforeheatmap]: ./output_images/before_heatmap.png
[rawheatmap]: ./output_images/raw_heatmap.png
[labelheatmap]: ./output_images/label_heatmap.png
[afterheatmap]: ./output_images/after_heatmap.png


## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the 4th code cell of the IPython notebook 

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and settled on the following:

orientation = 9  
pix_per_cell = 8  
cell_per_block = 2  

These HOG parameters were optimal as discovered through trial & error. 
The cell_per_block and pixel_per_cell values were motivated by the fact that car images are highly recognizable even when at a small resolution. Finally, an orientation bin of 9 was general enough to capture the orientation details for car images. A too high orientation value would make HOGs from car images taken at different angles too distinct from each other.


#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The code for this step is contained in the 4th code cell of the IPython notebook.
See the method, `extract_features(..)`. 

Images of cars and not-cars were split into a training & testing set (80%, 20%). A stratified split was performed, to ensure that each class is well represented. 

After which, each image was converted to YCrCb. The spatial features of the image were extracted by resizing the image to 32x32, and then stacking the pixel intensities of each channel into a vector.  
 
A color histogram, of 32 bins was also used on each channel, and the histogram is stacked into a vector. Finally, HOG information was also transformed into a vector. All these vectors were concatenated into one long feature vector, and normalized. This vector represents our input image.

The long feature vector was fed into a linear svm. The linear svm performs much faster than an svm using a rbf kernel. This speed factor is crucial for the sliding window search that comes in the next section. Consequently, the linear svm was chosen for it's speed.

The classifier had a test accuracy of about 99.12%, which is high enough given the relatively fast linear svm algorithm.


### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The sliding window search was implemented in code cell 15 of the IPython notebook, in the `mark_cars(..)` function. 

For each input scale, we resize the input image by dividing it's height & width by that scale.
We also crop the image by a given y-dimension range, because cars only appear on the road. This reduces the number of windows, thus reducing the time taken for processing. 

The image is split into square cells of 8 pixels. Each window is made of 64 pixels. Each window is paced with a step size of 2 cells from each other, meaning that there is overlapping present. This means that each window overlaps by 25%. This value is sufficient as higher overlaps results in a loss of information. Lower overlaps would however, require a lot more computation since there are more windows in the image.

The scales to search were empirically determined, by observing the size of cars in the project video. Each window should contain sufficient car features, similar to the training images. The scale sizes used were 1, 1.5, 2 and 2.5, the smaller vaules for catching cars further away and larger values for catching cars relatively close to the driver.

![alt text][regionandscales]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on four scales (scales mentioned previously) using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here is an example image:

![alt text][beforeheatmap]

Performance was optimized, by limiting the windows to only regions where cars can be found. Furthermore, the larger scale searches will only be performed on regions that are closer to the vehicle driven. This is because cars shrink as they are further away from the vehicle driven. A large scale will not be able to catch any cars that are far away and appear small.

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./output_video1.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

I used a running buffer of 8 images. If the mean heatmap of the buffer is within a threshold, we conclude that there is indeed a car detected, and create a bounding box around the heatmap region of the 8 images.

This code is present in code cell 17 of the IPython notebook.

Here's an example result showing a combined heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video.

### Combined Heatmap

![alt text][rawheatmap]

### Labeled Heatmap

![alt text][labelheatmap]

### Creating a bounding box with the Heatmap

![alt text][afterheatmap]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Problems faced was definitely computational time. I used 4 scales to increase accuracy of the algorithm. This is because cars on different lanes away from the camera will appear at a different size. Consequently, we do have to be careful and use more scales to capture all cars.

The pipeline is likely to fail when it rains, or when there's snow. Low light conditions may also be a problem. A simple way to make this more robust is to use up and coming deep learning algorithms like YOLO. YOLO has been noted to be extremely accurate, while requiring less computation as compared to other deep learning techniques. YOLO can also detect non-cars, such as people and bicycles, which is definitely another problem that the existing svm classifier will struggle with.