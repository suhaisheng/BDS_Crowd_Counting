from __future__ import print_function
import json
# f = open("Settings.json", encoding='utf-8')
from matplotlib.patches import Ellipse, Circle
import matplotlib.pyplot as plt
import cv2
import numpy as np

# vis = True # False True
vis = True # visualization pictures
vis_IR = True # visualization Ignore points
vis_dot = False # visualization bbox as dots (bbox -> dots)
def _vis_img_rois(image, rois):
	print('shape: ', image.shape)
	plt.imshow(image)
	    # print 'class: ', cls, ' overlap: ', overlaps[i]
	for roi in rois:
		plt.gca().add_patch(
			plt.Rectangle((roi[0], roi[1]), roi[2],
			roi[3], fill=False,
			edgecolor='g', linewidth=2)
			)    
	plt.show()
	cv2.waitKey(1000)
	# print('done')

def _vis_img_dots(image, dots, color,radius=7.5):
	print('shape: ', image.shape)
	plt.imshow(image)
	    # print 'class: ', cls, ' overlap: ', overlaps[i]
	    # cir1 = Circle(xy = (0.0, 0.0), radius=2, alpha=0.5)
	# ax.add_patch(ell1)
	# ax.add_patch(cir1)
	for dot in dots:
		cir1 = Circle(xy = (dot[0], dot[1]), radius=radius, color=color, alpha=1)
		plt.gca().add_patch(cir1)

	plt.show()
	cv2.waitKey(1000)
	# print('done')

def _draw_hist(myList,Title,Xlabel,Ylabel,Xmin,Xmax,Ymin,Ymax):
	plt.hist(myList,100)
	plt.xlabel(Xlabel)
	plt.xlim(Xmin,Xmax)
	plt.ylabel(Ylabel)
	plt.ylim(Ymin,Ymax)
	plt.title(Title)
	plt.show()
root = '/data1/ShengGe/ZSQ/crowdcount-mcnn'

f = open(root + "/images/baidu_star_2018_train_stage1/baidu_star_2018/annotation/annotation_train_stage1.json")   
annotations = json.load(f)
f_stage2 = open(root + "/images/baidu_star_2018_train_stage2/annotation/annotation_train_stage2.json")  #new_anno_xy_transfered_576_stage2.json
annotations_stage2 = json.load(f_stage2)
#anno = annotations['annotations'] + annotations_stage2['annotations']   # combine
anno = annotations_stage2['annotations']
image_num = len(anno)  # 2859
print("image_num:%d"%(image_num))

base_image_path = root + "/images/baidu_star_2018_train_stage2/image/"
base_image_path_stage2 = root +"/images/baidu_star_2018_train_stage1/baidu_star_2018/image/"
# cv2.namedWindow("Image")  
#image_name = base_image_path + anno[0]['name']
# im = cv2.imread(image_name)
# cv2.imshow("Image", im)


# ignore_region
dot_num = 0
rec_num = 0
IR = 0

nums = []
n = 0

for i in range(1342, image_num):  # 3625 +

	print('refer image %s'%(anno[i]['id']))

	image_name = base_image_path + anno[i]['name']
	anno_type = anno[i]['type']

	# if vis:
	img = cv2.imread(image_name)
	# shape_o = img.shape
	# img = cv2.resize(img, (1024, 512))
	
	# h, w = shape_o[0], shape_o[1]
	# ratio_h = h / 512.
	# ratio_w = w / 1024.
	ratio_h = 1.
	ratio_w = 1.
	# print(ratio_h, ' ', ratio_h)
	# xy["y"] = int(gt[i][1] / ratio_h)
	# xy["x"] = int(gt[i][0] / ratio_w)	
	# img = cv2.imread(image_name)
	# shape = img.shape
	# if shape[0] > max_shape[0]:
	# 	max_shape[0] = shape[0]
	# if shape[1] > max_shape[1]:
	# 	max_shape[1] = shape[1]
	# if shape[0] == 2560 or shape[1] == 2560:
	# 	n+=1
	# 	print(shape)
	# 	print(i)
	# 	if not (shape[0] == 1920 and shape[1] == 2560):
	# 		print(error)
	# if shape[0] == 2048 and shape[1] == 2048:
	# 	print(i)
	# 	break
	# if i == 3409:
	# 	img = cv2.imread(image_name)
	# 	print(img.shape)
	# 	plt.imshow(img)

	# img = cv2.imread(image_name)
	# print(image)
	# print('shape: ', img.shape)

	nums.append(anno[i]['num'])
	# nums = int(nums)
	
	has_IR = False
	if vis_IR:
		# img = cv2.imread(image_name)
		print('Vis Ignore Region')
		if len(anno[i]['ignore_region']):
			IR += 1
			has_IR = True
			print("ignore_region_num:",len(anno[i]['ignore_region']))
			print(len(anno[i]['ignore_region'][0]))
			print("crowd count:",anno[i]['num'], len(anno[i]['annotation']))
			for ignore_region_id in range(len(anno[i]['ignore_region'])):	
				dots_IR = []
				for annotation_id in range(len(anno[i]['ignore_region'][ignore_region_id])):
					annotation_item = anno[i]['ignore_region'][ignore_region_id][annotation_id]
					dot_IR = [annotation_item['x'], annotation_item['y']]
					dots_IR.append(dot_IR)
				#cv2.polylines(img, np.array(dots_IR), 1,255)
				origin_img = img
				cv2.fillPoly(img, [np.array(dots_IR)], 127)

				result_img = np.hstack((img, origin_img))
    			#result_img = result_img.astype(np.uint8, copy=False)

				plt.imshow(result_img)
				# print(len(dots_IR))
				# for j in range(len(dots_IR)-1):
				# 	plt.plot(dots_IR[j], dots_IR[j+1], linewidth=2.0)
				plt.show()
				# cv2.waitKey(1000)
				_vis_img_dots(img, dots_IR, 'g')	

	head_size = 30
	print(anno_type)
	if anno_type == 'bbox':
		rec_num += 1
		print('Vis Bbox')
		if vis:
			rois = [];
			for annotation_id in range(len(anno[i]['annotation'])):
				annotation_item = anno[i]['annotation'][annotation_id]
				roi = [annotation_item['x'], annotation_item['y'], annotation_item['w'], annotation_item['h']]
				# resize
				roi[0],roi[1],roi[2],roi[3] = roi[0]/ratio_w, roi[1]/ratio_h, roi[2]/ratio_w, roi[3]/ratio_h

				# x1 = roi[0] + 1/2.*roi[2] - head_size / 2
				# y1 = roi[1] + 1/4.*roi[2] - head_size / 2
				# w1 = head_size
				# h1 = head_size

				# head_bbox = [x1, y1, w1, h1]

				rois.append(roi)
			_vis_img_rois(img, rois)	

		if vis_dot:
			print('Vis Bbox transfer to dot')
			# img = cv2.imread(image_name)
			dot_ts = [];
			for annotation_id in range(len(anno[i]['annotation'])):
				annotation_item = anno[i]['annotation'][annotation_id]
				roi = [annotation_item['x'], annotation_item['y'], annotation_item['w'], annotation_item['h']]
				dot_t = [roi[0] + roi[2]/2., roi[1] + roi[2] / 4.] 

				dot_ts.append(dot_t)
			_vis_img_dots(img, dot_ts, radius=7.5)		

	elif anno_type == 'dot':
		dot_num += 1
		print('Vis Dot')
		if vis:
			dots = []
			hbbox = []
			for annotation_id in range(len(anno[i]['annotation'])):
				annotation_item = anno[i]['annotation'][annotation_id]
				dot = [annotation_item['x'], annotation_item['y']]
				dot[0],dot[1] = dot[0]/ratio_w, dot[1]/ratio_h
				# x1 = dot[0] - head_size/2.
				# y1 = dot[1] - head_size/2.
				# w1 = head_size
				# h1 = head_size
				# hbbox.append([x1, y1, w1, h1])
				dots.append(dot)
			_vis_img_dots(img, dots, 'r')
			# _vis_img_rois(img, hbbox)	


# print(max_shape)
# print('n: ', n)
    # cv2.imshow("Image", img)
# _draw_hist(nums,'numDIS','number','fre',0,600,0.0,1000)
#     # cv2.waitKey(10000)
# print("IRnum: ", IR)
# print("dot_num: ", dot_num, ' rec_num: ', rec_num)


# cv2.destroyAllWindows()  