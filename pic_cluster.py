from __future__ import print_function
import json
# f = open("Settings.json", encoding='utf-8')
from matplotlib.patches import Ellipse, Circle
import matplotlib.pyplot as plt
# import cv2

import numpy as np
import cv2

from sklearn.cluster import KMeans
from sklearn.externals import joblib
from sklearn import cluster
import os
import os.path 
from skimage import feature as ft
import shutil

def mkdirs(dir_name):
	if not os.path.exists(dir_name):
		os.makedirs(dir_name)


def get_rgb_hist_feature(image_name):
	hist_feature = []
	img = cv2.imread(image_name)
	color = ('b', 'g', 'r')
	for index, value in enumerate(color):
		hist = cv2.calcHist([img], [index], None, [256], [0, 256])
		cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
		hist_feature.append(hist)
	return np.array(hist_feature).ravel()


def get_hsv_hist_feature(image_name):
	hist_feature = []
	img = cv2.imread(image_name)
	img = cv2.resize(img, (256, 256))
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

	hist_h = cv2.calcHist([hsv], [0], None, [180], [0, 180])
	hist_h = cv2.normalize(hist_h, hist_h)
	hist_feature[0:180] = hist_h[:,0]

	hist_s = cv2.calcHist([hsv], [1], None, [256], [0, 256])
	hist_s = cv2.normalize(hist_s, hist_s)
	hist_feature[180:436] = hist_s[:,0]

	hist_v = cv2.calcHist([hsv], [2], None, [256], [0, 256])
	hist_v = cv2.normalize(hist_v,hist_v)
	hist_feature[436:692] = hist_v[:,0]

	return np.array(hist_feature).ravel()


def get_hog_feature(image_name):
	img = cv2.imread(image_name)
	img = cv2.resize(img, (256, 256))
	gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	hog_feature = ft.hog(gray_image, orientations=8, pixels_per_cell=(16, 16),
								cells_per_block=(1, 1))
	return hog_feature.ravel()


# def main():

f = open("/data0/zsq/Research_2/CrowdCounting/crowdcount-mcnn/images/baidu_star_2018_train_stage1/baidu_star_2018/annotation/annotation_train_stage1.json")

anno = json.load(f)

f.close()
# print(anno["annotations"][0]['num'])

image_num = len(anno["annotations"])

# print("aa" + "/sssd")
base_image_path = "/data0/zsq/Research_2/CrowdCounting/crowdcount-mcnn/images/baidu_star_2018_train_stage1/baidu_star_2018/image/"
# cv2.namedWindow("Image")  
# image_name = base_image_path + anno["annotations"][0]['name']
n_clusters = 13

clusters_dirs = []
for idx in range(n_clusters):
	dir_name = '/data0/zsq/Research_2/CrowdCounting/crowdcount-mcnn/images/baidu_star_2018_train_stage1/baidu_star_2018/clusters/' + 'cluster_%02d/'%(idx)
	mkdirs(dir_name)
	clusters_dirs.append(dir_name)



features = []
for idx in range(image_num): # image_num
	if idx % 100 == 0:
		print('extract img %06d features'%(idx))
	image_name = base_image_path + anno["annotations"][idx]['name']
	# img = cv2.imread(image_name)
	feature = get_hog_feature(image_name)
	# feature = feature[np.newaxis, :]
	features.append(feature)
	# print(len(feature))
	# print(feature.shape)
features_npy = np.array(features)

k_means_estimator = KMeans(n_clusters = n_clusters)
# data = np.random.rand(10, 3)
print('clustering...')
res = k_means_estimator.fit_predict(features_npy)
label_pred = k_means_estimator.labels_
centroids = k_means_estimator.cluster_centers_
inertia = k_means_estimator.inertia_

for idx in range(image_num):
	if idx % 100 == 0:
		print('copy file %06d'%(idx))
	
	cluster_id = label_pred[idx]
	image_name = base_image_path + anno["annotations"][idx]['name']
	target_name = clusters_dirs[cluster_id] + '/%06d.jpg'%(anno["annotations"][idx]['id'])
	shutil.copyfile(image_name, target_name)




# if __name__ == '__main__':
# 	main
