import argparse
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import scipy.io
import scipy.misc
import numpy as np
import pandas as pd
import h5py
import PIL
import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Lambda, Conv2D
from keras.models import load_model, Model
from yolo_utils import read_classes, read_anchors, generate_colors, preprocess_image, draw_boxes, scale_boxes
from yad2k.models.keras_yolo import yolo_head, yolo_boxes_to_corners, preprocess_true_boxes, yolo_loss, yolo_body

'''
# Important Note: As you can see, we import Keras's backend as K. This means that to use a Keras function in this notebook,
# you will need to write: K.function(...).



# If you have 80 classes that you want the object detector to recognize, you can represent the class label $c$ either as an integer from 1 to 80,
# or as an 80-dimensional vector (with 80 numbers) one component of which is 1 and the rest of which are 0. The video lectures had used the
# latter representation; in this notebook, we will use both representations, depending on which is more convenient for a particular step.

# In this exercise, you will learn how "You Only Look Once" (YOLO) performs object detection, and then apply it to car detection. Because
# the YOLO model is very computationally expensive to train, we will load pre-trained weights for you to use.

# 2 - YOLO
# "You Only Look Once" (YOLO) is a popular algorithm because it achieves high accuracy while also being able to run in real-time.
# This algorithm "only looks once" at the image in the sense that it requires only one forward propagation pass through the network to make predictions.
# After non-max suppression, it then outputs recognized objects together with the bounding boxes.
#
# 2.1 - Model details
# Inputs and outputs
# The input is a batch of images, and each image has the shape (m, 608, 608, 3)
# The output is a list of bounding boxes along with the recognized classes. Each bounding box is represented by 6
# numbers $(p_c, b_x, b_y, b_h, b_w, c)$ as explained above. If you expand $c$ into an 80-dimensional vector, each bounding box
# is then represented by 85 numbers.
# Anchor Boxes
# Anchor boxes are chosen by exploring the training data to choose reasonable height/width ratios that represent the different classes.
# For this assignment, 5 anchor boxes were chosen for you (to cover the 80 classes), and stored in the file './model_data/yolo_anchors.txt'
# The dimension for anchor boxes is the second to last dimension in the encoding: $(m, n_H,n_W,anchors,classes)$.
# The YOLO architecture is: IMAGE (m, 608, 608, 3) -> DEEP CNN -> ENCODING (m, 19, 19, 5, 85).


# 2.2 - Filtering with a threshold on class scores
# You are going to first apply a filter by thresholding. You would like to get rid of any box for which the class "score" is less than a chosen threshold.
#
# The model gives you a total of 19x19x5x85 numbers, with each box described by 85 numbers. It is convenient to rearrange the (19,19,5,85)
# (or (19,19,425)) dimensional tensor into the following variables:
#
# box_confidence: tensor of shape $(19 \times 19, 5, 1)$ containing $p_c$ (confidence probability that there's some object) for each
# of the 5 boxes predicted in each of the 19x19 cells.
# boxes: tensor of shape $(19 \times 19, 5, 4)$ containing the midpoint and dimensions $(b_x, b_y, b_h, b_w)$ for each of the 5 boxes in each cell.
# box_class_probs: tensor of shape $(19 \times 19, 5, 80)$ containing the "class probabilities" $(c_1, c_2, ... c_{80})$ for each of the 80
# classes for each of the 5 boxes per cell.

# Compute box scores by doing the elementwise product as described in Figure 4 ($p \times c$).
# The following code may help you choose the right operator:
#
# a = np.random.randn(19*19, 5, 1)
# b = np.random.randn(19*19, 5, 80)
# c = a * b # shape of c will be (19*19, 5, 80)
# This is an example of broadcasting (multiplying vectors of different sizes).
#
# For each box, find:
#
# the index of the class with the maximum box score
# the corresponding box score
#
# Useful references
#
# Keras argmax
# Keras max
# Additional Hints
#
# For the axis parameter of argmax and max, if you want to select the last axis, one way to do so is to set axis=-1. This is similar to Python
# array indexing, where you can select the last position of an array using arrayname[-1].
# Applying max normally collapses the axis for which the maximum is applied. keepdims=False is the default option, and allows that dimension to be removed.
# We don't need to keep the last dimension after applying the maximum here.
# Even though the documentation shows keras.backend.argmax, use keras.argmax. Similarly, use keras.max.
# Create a mask by using a threshold. As a reminder: ([0.9, 0.3, 0.4, 0.5, 0.1] < 0.4) returns: [False, True, False, False, True]. The mask should be
# True for the boxes you want to keep.
#
# Use TensorFlow to apply the mask to box_class_scores, boxes and box_classes to filter out the boxes we don't want. You should be left with just the
# subset of boxes you want to keep.
'''
def yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold=.6):
    """Filters YOLO boxes by thresholding on object and class confidence.

    Arguments:
    box_confidence -- tensor of shape (19, 19, 5, 1)
    boxes -- tensor of shape (19, 19, 5, 4)
    box_class_probs -- tensor of shape (19, 19, 5, 80)
    threshold -- real value, if [ highest class probability score < threshold], then get rid of the corresponding box

    Returns:
    scores -- tensor of shape (None,), containing the class probability score for selected boxes
    boxes -- tensor of shape (None, 4), containing (b_x, b_y, b_h, b_w) coordinates of selected boxes
    classes -- tensor of shape (None,), containing the index of the class detected by the selected boxes

    Note: "None" is here because you don't know the exact number of selected boxes, as it depends on the threshold.
    For example, the actual output size of scores would be (10,) if there are 10 boxes.
    """

    # Step 1 : compute box scores
    box_scores = box_confidence * box_class_probs

    # Step 2: Find the box_classes using the max box_scores, keep track of the corresponding score
    box_classes = K.argmax(box_scores, axis=-1)
    box_classes_scores = K.max(box_scores, axis=-1)
    # print(box_classes.shape, box_classes_scores.shape)

    # Step 3: Create a filtering mask based on "box_class_scores" by using "threshold".
    # The mask should have the
    # same dimension as box_class_scores, and be True for the boxes you want to keep
    # (with probability >= threshold)
    filtering_mask = (box_classes_scores >= threshold)

    # Step 4: Apply the mask to box_class_scores, boxes and box_classes
    scores = tf.boolean_mask(box_classes_scores, filtering_mask)
    boxes =  tf.boolean_mask(boxes, filtering_mask)
    classes = tf.boolean_mask(box_classes, filtering_mask)

    return scores, boxes, classes

#
with tf.compat.v1.Session() as test_a:
    box_confidence = tf.compat.v1.random_normal([19, 19, 5, 1], mean=1, stddev=4, seed = 1)
    boxes = tf.compat.v1.random_normal([19, 19, 5, 4], mean=1, stddev=4, seed = 1)
    box_class_probs = tf.compat.v1.random_normal([19, 19, 5, 80], mean=1, stddev=4, seed = 1)
    scores, boxes, classes = yolo_filter_boxes(box_confidence, boxes, box_class_probs,
                                               threshold = 0.5)
    print("scores[2] = " + str(scores[2].eval()))
    print("boxes[2] = " + str(boxes[2].eval()))
    print("classes[2] = " + str(classes[2].eval()))
    print("scores.shape = " + str(scores.shape))
    print("boxes.shape = " + str(boxes.shape))
    print("classes.shape = " + str(classes.shape))
'''
# In this code, we use the convention that (0,0) is the top-left corner of an image, (1,0) is the upper-right corner, and (1,1) is the lower-right corner. In other words,
# the (0,0) origin starts at the top left corner of the image. As x increases, we move to the right. As y increases, we move down.
# For this exercise, we define a box using its two corners: upper left $(x_1, y_1)$ and lower right $(x_2,y_2)$, instead of using the midpoint, height and width.
# (This makes it a bit easier to calculate the intersection).
# To calculate the area of a rectangle, multiply its height $(y_2 - y_1)$ by its width $(x_2 - x_1)$. (Since $(x_1,y_1)$ is the top left and $x_2,y_2$ are
# the bottom right, these differences should be non-negative.
# To find the intersection of the two boxes $(xi_{1}, yi_{1}, xi_{2}, yi_{2})$:
# Feel free to draw some examples on paper to clarify this conceptually.
# The top left corner of the intersection $(xi_{1}, yi_{1})$ is found by comparing the top left corners $(x_1, y_1)$ of the two boxes and finding a vertex that
# has an x-coordinate that is closer to the right, and y-coordinate that is closer to the bottom.
# The bottom right corner of the intersection $(xi_{2}, yi_{2})$ is found by comparing the bottom right corners $(x_2,y_2)$ of the two boxes and finding a vertex
# whose x-coordinate is closer to the left, and the y-coordinate that is closer to the top.
# The two boxes may have no intersection. You can detect this if the intersection coordinates you calculate end up being the top right and/or bottom left corners
# of an intersection box. Another way to think of this is if you calculate the height $(y_2 - y_1)$ or width $(x_2 - x_1)$ and find that at least one of these
# lengths is negative, then there is no intersection (intersection area is zero).
# The two boxes may intersect at the edges or vertices, in which case the intersection area is still zero. This happens when either the height or width (or both)
# of the calculated intersection is zero.


# xi1 = maximum of the x1 coordinates of the two boxes
# yi1 = maximum of the y1 coordinates of the two boxes
# xi2 = minimum of the x2 coordinates of the two boxes
# yi2 = minimum of the y2 coordinates of the two boxes
# inter_area = You can use max(height, 0) and max(width, 0)
'''
def iou(box1, box2):
    """Implement the intersection over union (IoU) between box1 and box2

    Arguments:
    box1 -- first box, list object with coordinates (box1_x1, box1_y1, box1_x2, box_1_y2)
    box2 -- second box, list object with coordinates (box2_x1, box2_y1, box2_x2, box2_y2)
    """

    # Assign variable names to coordinates for clarity
    (box1_x1, box1_y1, box1_x2, box1_y2) = box1
    (box2_x1, box2_y1, box2_x2, box2_y2) = box2

    # Calculate the (yi1, xi1, yi2, xi2) coordinates of the intersection of box1
    # and box2. Calculate its Area.
    xi1 = np.maximum(box1_x1, box2_x1)
    yi1 = np.maximum(box1_y1, box2_y1)
    xi2 = np.minimum(box1_x2, box2_x2)
    yi2 = np.minimum(box1_y2, box2_y2)

    inter_width = xi2-xi1
    inter_height = yi2-yi1

    inter_area = max(inter_height, 0) * max(inter_width, 0)

    # Calculate the Union area by using Formula: Union(A,B) = A + B - Inter(A,B)
    box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
    box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
    union_area = box1_area + box2_area - inter_area

    iou = float(inter_area) / float(union_area)

    return iou

box1 = (2, 1, 4, 3)
box2 = (1, 2, 3, 4)
print("iou for intersecting boxes = " + str(iou(box1, box2)))

## Test case 2: boxes do not intersect
box1 = (1,2,3,4)
box2 = (5,6,7,8)
print("iou for non-intersecting boxes = " + str(iou(box1,box2)))

## Test case 3: boxes intersect at vertices only
box1 = (1,1,2,2)
box2 = (2,2,3,3)
print("iou for boxes that only touch at vertices = " + str(iou(box1,box2)))

## Test case 4: boxes intersect at edge only
box1 = (1,1,3,3)
box2 = (2,3,3,4)
print("iou for boxes that only touch at edges = " + str(iou(box1,box2)))


def yolo_non_max_suppression(scores, boxes, classes, max_boxes=10, iou_threshold=0.5):
    """
    Applies Non-max suppression (NMS) to set of boxes

    Arguments:
    scores -- tensor of shape (None,), output of yolo_filter_boxes()
    boxes -- tensor of shape (None, 4), output of yolo_filter_boxes() that have been scaled to the image size (see later)
    classes -- tensor of shape (None,), output of yolo_filter_boxes()
    max_boxes -- integer, maximum number of predicted boxes you'd like
    iou_threshold -- real value, "intersection over union" threshold used for NMS filtering

    Returns:
    scores -- tensor of shape (, None), predicted score for each box
    boxes -- tensor of shape (4, None), predicted box coordinates
    classes -- tensor of shape (, None), predicted class for each box

    Note: The "None" dimension of the output tensors has obviously to be less than max_boxes. Note also that this
    function will transpose the shapes of scores, boxes, classes. This is made for convenience.
    """
    # Use tf.image.non_max_suppression() to get the list of indices corresponding to boxes you keep
    nms_indicies = tf.image.non_max_suppression(boxes=boxes, scores=scores, max_output_size=max_boxes, iou_threshold=iou_threshold)

    # Use K.gather() to select only nms_indices from scores, boxes and classes
    scores = K.gather(scores, nms_indicies)
    boxes = K.gather(boxes, nms_indicies)
    classes = K.gather(classes, nms_indicies)

    return scores, boxes, classes

with tf.compat.v1.Session() as test_b:
    scores = tf.compat.v1.random_normal([54,], mean=1, stddev=4, seed = 1)
    boxes = tf.compat.v1.random_normal([54, 4], mean=1, stddev=4, seed = 1)
    classes = tf.compat.v1.random_normal([54,], mean=1, stddev=4, seed = 1)
    scores, boxes, classes = yolo_non_max_suppression(scores, boxes, classes)
    print("scores[2] = " + str(scores[2].eval()))
    print("boxes[2] = " + str(boxes[2].eval()))
    print("classes[2] = " + str(classes[2].eval()))
    print("scores.shape = " + str(scores.eval().shape))
    print("boxes.shape = " + str(boxes.eval().shape))
    print("classes.shape = " + str(classes.eval().shape))



# GRADED FUNCTION: yolo_eval
'''
It's time to implement a function taking the output of the deep CNN (the 19x19x5x85 dimensional encoding) and filtering through all the boxes using the functions 
you've just implemented.

Exercise: Implement yolo_eval() which takes the output of the YOLO encoding and filters the boxes using score threshold and NMS. There's just one last 
implementational detail you have to know. There're a few ways of representing boxes, such as via their corners or via their midpoint and height/width. 
YOLO converts between a few such formats at different times, using the following functions (which we have provided):

boxes = yolo_boxes_to_corners(box_xy, box_wh)
which converts the yolo box coordinates (x,y,w,h) to box corners' coordinates (x1, y1, x2, y2) to fit the input of yolo_filter_boxes

boxes = scale_boxes(boxes, image_shape)
YOLO's network was trained to run on 608x608 images. If you are testing this data on a different size image--for example, 
the car detection dataset had 720x1280 images--this step rescales the boxes so that they can be plotted on top of the original 720x1280 image.

Don't worry about these two functions; we'll show you where they need to be called.
'''
def yolo_eval(yolo_outputs, image_shape=(720., 1280.), max_boxes=10, score_threshold=.6, iou_threshold=.5):
    """
    Converts the output of YOLO encoding (a lot of boxes) to your predicted boxes along with their scores, box coordinates and classes.

    Arguments:
    yolo_outputs -- output of the encoding model (for image_shape of (608, 608, 3)), contains 4 tensors:
                    box_confidence: tensor of shape (None, 19, 19, 5, 1)
                    box_xy: tensor of shape (None, 19, 19, 5, 2)
                    box_wh: tensor of shape (None, 19, 19, 5, 2)
                    box_class_probs: tensor of shape (None, 19, 19, 5, 80)
    image_shape -- tensor of shape (2,) containing the input shape, in this notebook we use (608., 608.) (has to be float32 dtype)
    max_boxes -- integer, maximum number of predicted boxes you'd like
    score_threshold -- real value, if [ highest class probability score < threshold], then get rid of the corresponding box
    iou_threshold -- real value, "intersection over union" threshold used for NMS filtering

    Returns:
    scores -- tensor of shape (None, ), predicted score for each box
    boxes -- tensor of shape (None, 4), predicted box coordinates
    classes -- tensor of shape (None,), predicted class for each box
    """

    # Retrieve outputs of the YOLO model (≈1 line)
    box_confidence, box_xy, box_wh, box_class_probs = yolo_outputs

    # Convert boxes to be ready for filtering functions (convert boxes box_xy and box_wh to corner coordinates)
    boxes = yolo_boxes_to_corners(box_xy, box_wh)

    # Use one of the functions you've implemented to perform Score-filtering with a threshold of score_threshold
    scores, boxes, classes = yolo_filter_boxes(box_confidence, boxes, box_class_probs, score_threshold)

    # Scale boxes back to original image shape.
    boxes = scale_boxes(boxes, image_shape)

    # Use one of the functions you've implemented to perform Non-max suppression with
    # maximum number of boxes set to max_boxes and a threshold of iou_threshold (≈1 line)
    scores, boxes, classes = yolo_non_max_suppression(scores, boxes, classes, max_boxes, iou_threshold)

    return scores, boxes, classes


with tf.compat.v1.Session() as test_b:
    yolo_outputs = (tf.compat.v1.random_normal([19, 19, 5, 1], mean=1, stddev=4, seed = 1),
                    tf.compat.v1.random_normal([19, 19, 5, 2], mean=1, stddev=4, seed = 1),
                    tf.compat.v1.random_normal([19, 19, 5, 2], mean=1, stddev=4, seed = 1),
                    tf.compat.v1.random_normal([19, 19, 5, 80], mean=1, stddev=4, seed = 1))
    scores, boxes, classes = yolo_eval(yolo_outputs)
    print("scores[2] = " + str(scores[2].eval()))
    print("boxes[2] = " + str(boxes[2].eval()))
    print("classes[2] = " + str(classes[2].eval()))
    print("scores.shape = " + str(scores.eval().shape))
    print("boxes.shape = " + str(boxes.eval().shape))
    print("classes.shape = " + str(classes.eval().shape))
'''
Summary for YOLO:¶
Input image (608, 608, 3)
The input image goes through a CNN, resulting in a (19,19,5,85) dimensional output.
After flattening the last two dimensions, the output is a volume of shape (19, 19, 425):
Each cell in a 19x19 grid over the input image gives 425 numbers.
425 = 5 x 85 because each cell contains predictions for 5 boxes, corresponding to 5 anchor boxes, as seen in lecture.
85 = 5 + 80 where 5 is because $(p_c, b_x, b_y, b_h, b_w)$ has 5 numbers, and 80 is the number of classes we'd like to detect
You then select only few boxes based on:
Score-thresholding: throw away boxes that have detected a class with a score less than the threshold
Non-max suppression: Compute the Intersection over Union and avoid selecting overlapping boxes
This gives you YOLO's final output.
'''

sess = tf.compat.v1.keras.backend.get_session()
class_names = read_classes("model_data\\coco_classes.txt")
anchors = read_anchors("model_data\\yolo_anchors.txt")
image_shape = (720., 1280.)

'''Loading a pre-trained model¶
Training a YOLO model takes a very long time and requires a fairly large dataset of labelled bounding boxes for a large range of target classes.
You are going to load an existing pre-trained Keras YOLO model stored in "yolo.h5".
These weights come from the official YOLO website, and were converted using a function written by Allan Zelener. References are at the end of this notebook. Technically, these are the parameters from the "YOLOv2" model, but we will simply refer to it as "YOLO" in this notebook.
Run the cell below to load the model from this file.
'''

yolo_model = load_model("model_data\\yolo.h5")
yolo_model.output
tf.keras.layers.Dropout