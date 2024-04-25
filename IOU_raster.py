# IOU "Intersection Over Union" Calculation
# Special Thanks to My lovely Deep Learning Expert : Hiroyuki Miyazaki, Ph.D. 
# Thepchai Srinoi
# Department of Survey Engineering Chulalongkorn University

import rasterio
from rasterio import features
from sklearn import metrics

# Firstly, convert polygon layer to raster 
if 1 :
    ground_truth_file = 'eng_building_ground.tiff'
    prediction_result_file = 'eng_windows_pred.tiff'

if 0 :
    ground_truth_file = 'sci_building_ground.tiff'
    prediction_result_file = 'sci_windows_pred.tiff'


prediction_result = rasterio.open(prediction_result_file)
ground_truth_rasterized = rasterio.open(ground_truth_file)

ground_truth_rasterized_1d_array = ground_truth_rasterized.read(1).ravel()
prediction_result_1d_array = prediction_result.read(1).ravel()

confusion_matrix = metrics.confusion_matrix(ground_truth_rasterized_1d_array, prediction_result_1d_array)
iou =  confusion_matrix[1,1] / (confusion_matrix[1,1] + confusion_matrix[0,1] + confusion_matrix[1,0])
print("IoU:       %.4f" % iou)

print("Accuracy:  %.4f" % metrics.accuracy_score(ground_truth_rasterized_1d_array, prediction_result_1d_array))
print("Precision: %.4f" % metrics.precision_score(ground_truth_rasterized_1d_array, prediction_result_1d_array))
print("Recall:    %.4f" % metrics.recall_score(ground_truth_rasterized_1d_array, prediction_result_1d_array))

import pdb; pdb.set_trace()

'''
Engineering
GeoSAM
IoU:       0.9064
Accuracy:  0.9575
Precision: 0.9829
Recall:    0.9209

(Pdb) confusion_matrix
array([[12839731,   168660],
       [  831064,  9679564]], dtype=int64)
Google

IoU:       0.6296
Accuracy:  0.8151
Precision: 0.8571
Recall:    0.7034
(Pdb) confusion_matrix
array([[11776109,  1232282],
       [ 3117081,  7393547]], dtype=int64)
       
Windows
IoU:       0.7265
Accuracy:  0.8620
Precision: 0.8641
Recall:    0.8202
(Pdb) confusion_matrix
array([[11653124,  1355267],
       [ 1889893,  8620735]], dtype=int64)

'''