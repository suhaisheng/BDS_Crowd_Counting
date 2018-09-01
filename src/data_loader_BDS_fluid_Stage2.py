import numpy as np
import cv2
import os
import random
import json
import pandas as pd
import paddle.v2 as paddle
# import matplotlib.pyplot as plt

def _im_list_to_blob(ims):
    """Convert a list of images into a network input.

    Assumes images are already prepared (means subtracted, BGR order, ...).
    """
    max_shape = np.array([im.shape for im in ims]).max(axis=0)
    num_images = len(ims)
    blob = np.zeros((num_images, 3, max_shape[0], max_shape[1]),
                    dtype=np.float32)

    for i in xrange(num_images):
        im = ims[i]
        im = np.transpose(im, (2,0,1))  # CHW

        start_y = int(np.floor((max_shape[0] - im.shape[1]) / 2.))
        start_x = int(np.floor((max_shape[1] - im.shape[2]) / 2.))

        blob[i, :, start_y:start_y+im.shape[1], start_x:start_x+im.shape[2]] = im # batch method 
    # print('blob.shape: ', blob.shape)
    return blob

def _den_list_to_blob(ims):
    """Convert a list of images into a network input.

    Assumes images are already prepared (means subtracted, BGR order, ...).
    """
    max_shape = np.array([im.shape for im in ims]).max(axis=0)
    num_images = len(ims)
    blob = np.zeros((num_images, 1, max_shape[0], max_shape[1]),
                    dtype=np.float32)

    for i in xrange(num_images):
        im = ims[i]
        start_y = int(np.floor((max_shape[0] - im.shape[0]) / 2.))
        start_x = int(np.floor((max_shape[1] - im.shape[1]) / 2.))

        blob[i, 0, start_y:start_y+im.shape[0], start_x:start_x+im.shape[1]] = im # batch method 
    # print('blob.shape: ', blob.shape)
    return blob

def _clip_start_end_version_1(w, h, start_x, start_y, end_x, end_y):
    if start_x < 0: 
        start_x = 0
    if start_y < 0:
        start_y = 0

    if end_x > w:
        end_x = w
    if end_y > h:
        end_y = h

    ww = end_x - start_x 
    hh = end_y - start_y

    ww = (ww/4)*4
    hh = (hh/4)*4

    end_x = start_x + ww
    end_y = start_y + hh

    return int(start_x), int(start_y), int(end_x), int(end_y)


def _clip_start_end(w, h, start_x, start_y, end_x, end_y):
    crop_w_tmp = end_x - start_x
    crop_h_tmp = end_y - start_y

    if start_x < 0: 
        start_x = 0
        # end_x = start_x + crop_w_tmp
        end_x = 512
    if start_y < 0:
        start_y = 0
        # end_y = start_y + crop_h_tmp
        end_y = 288
    # if end_x >= w:
    #     end_x = w
        # start_x = end_x - crop_w_tmp
    if end_x >= 1024:
        end_x = 1024
        start_x = 512
    # if end_y >= h:
    #     end_y = h
    #     start_y = end_y - crop_h_tmp
    if end_y >= 576:
        end_y = 576
        start_y = 288
    # if end_x > w:
    #     end_x = w
    # if end_y > h:
    #     end_y = h

    # ww = end_x - start_x 
    # hh = end_y - start_y

    # ww = (ww/4)*4
    # hh = (hh/4)*4

    # end_x = start_x + ww
    # end_y = start_y + hh

    return int(start_x), int(start_y), int(end_x), int(end_y)


def _vis_minibatch(img):
    """Visualize a mini-batch for debugging."""
    import matplotlib.pyplot as plt
    plt.imshow(img)
    plt.show()


class ImageDataLoader():
    def __init__(self, root_path, batch_size = 4, training = True, testing = False, shuffle=True, gt_downsample=False, pre_load=False):
        #pre_load: if true, all training and validation images are loaded into CPU RAM for faster processing.
        #          This avoids frequent file reads. Use this only for small datasets.
        # 
        self.root_path = root_path
        #self.f = open(self.root_path + "/images/baidu_star_2018_train_stage1/baidu_star_2018/annotation/new_anno_xy_transfered_stage1.json")
        self.f = open(self.root_path + "/images/baidu_star_2018_train_stage1/baidu_star_2018/annotation/new_anno_xy_transfered_576_stage1.json")
        self.annotations = json.load(self.f)
        self.f.close()

        #self.f_stage2 = open(self.root_path + "/images/baidu_star_2018_train_stage2/annotation/new_anno_xy_transfered_stage2.json")
        self.f_stage2 = open(self.root_path + "/images/baidu_star_2018_train_stage2/annotation/new_anno_xy_transfered_576_stage2.json")
        self.annotations_stage2 = json.load(self.f_stage2)    
        self.f_stage2.close()

        self.f_test = open(self.root_path + "/images/baidu_star_2018_test_stage2/baidu_star_2018/annotation/annotation_test_stage2.json")
        # self.f_test_Stage2 = open(self.root_path + "/images/baidu_star_2018_test_stage2/baidu_star_2018/annotation/annotation_test_stage2.json")

        self.annotations_test = json.load(self.f_test)
        self.f_test.close()
        # self.f_test_Stage2.close()

        self.base_image_path = self.root_path + "images/baidu_star_2018_train_stage1/baidu_star_2018/image/"
        self.base_image_path_test = self.root_path + "images/baidu_star_2018_test_stage2/baidu_star_2018/image/"
        # self.
        self.training = training
        self.testing = testing 

        self.img_normalization_method = 'SubMean' # 'ImageNet' # 
        # self.density_gt_path = self.root_path + "/images/baidu_star_2018_train_stage1/baidu_star_2018/GT_Density/"
        # images are gray images
        # density maps are 1 channel
        self.img_npy_path = self.root_path + "/images/baidu_star_2018_train_stage1/baidu_star_2018/image_3_resize_576_npy/"
        self.den_npy_path = self.root_path + "/images/baidu_star_2018_train_stage1/baidu_star_2018/den_gt_adaptive/"

        self.annos_test = self.annotations_test['annotations']
        self.num_test_images = len(self.annos_test)
  
        self.annos = self.annotations['annotations'] + self.annotations_stage2['annotations']   # combine
        self.annos_origin = self.annotations['annotations'] + self.annotations_stage2['annotations']
        # self.annos_train = 
        # self.annos_val = 

        # self.train_ids = np.load("/home/fanglj/ZSQ/Research_2/CrowdCounting/crowdcount-mcnn/images/baidu_star_2018_train_stage1/baidu_star_2018/train_ids.npy")
        # self.val_ids = np.load("/home/fanglj/ZSQ/Research_2/CrowdCounting/crowdcount-mcnn/images/baidu_star_2018_train_stage1/baidu_star_2018/val_ids.npy")
        self.train_ids = np.load(self.root_path + "/images/baidu_star_2018_train_stage1/baidu_star_2018/Stage2/train_ids_9_1.npy").astype(np.int64)  
        self.val_ids = np.load(self.root_path + "/images/baidu_star_2018_train_stage1/baidu_star_2018/Stage2/val_ids_9_1.npy").astype(np.int64)

        self.num_images = len(self.annos)  # 3625+2859

        self.num_train_images = self.train_ids.shape[0]  # 2285
        self.num_val_images = self.val_ids.shape[0] #253
        print(self.num_val_images)
        self.sparse_thre = 50 # 40
        self.batch_size = batch_size 
        self.test_batch_size = 1
        # self.data_path = data_path
        # self.gt_path = gt_path
        self.gt_downsample = gt_downsample
        self.pre_load = pre_load

        # self.data_files = [filename for filename in os.listdir(data_path) \
                           # if os.path.isfile(os.path.join(data_path,filename))]
        # self.data_files.sort()
        self.shuffle = shuffle
        # if shuffle:
        #     random.seed(2468)

        self.num_samples = len(self.annos)

        # self.blob_list = {}        
        self.id_list = range(0,self.num_samples)
           
        self.target_h = 500
        self.target_w = 1000      
        self.samples_to_crop = 1
        self.samples_to_crop_large = 1
        self.samples_to_crop_small = 1
        self.patch_size_ratio = 2
        self.images_list = []
        self.dens_list = []

        self.points_list = []
        self.IR_list = []

        for idx in range(self.num_samples):
            anno_type = self.annos[idx]['type']
            # img_name = self.base_image_path + self.annos[idx]['name']
            ht = 576
            wd = 1024


            # if len(self.annos[idx]['ignore_region']):
            #     IR += 1
            #     has_IR = True
            #     print(len(self.annos[idx]['ignore_region'][0]))
            #     dots_IR = []
            #     for annotation_id in range(len(anno["annotations"][i]['ignore_region'][0])):
            #         annotation_item = anno["annotations"][i]['ignore_region'][0][annotation_id]
            #         dot_IR = [annotation_item['x'], annotation_item['y']]
            #         dots_IR.append(dot_IR)

            if anno_type == 'bbox':
                dots = [];
                print('error')
                for annotation_id in range(len(self.annos[idx]['annotation'])):
                    annotation_item = self.annos[idx]['annotation'][annotation_id]
                    roi = [annotation_item['x'], annotation_item['y'], annotation_item['w'], annotation_item['h']]
                    dot_t = [roi[0] + roi[2]/2., roi[1] + roi[3] / 4.] 
                    x, y = dot_t[0], dot_t[1]
                    if x > 0 and x < wd  and y > 0 and y < ht:
                        dots.append(dot_t)   
                if len(dots) == 0:
                    print "-------error---------"
                self.points_list.append(dots)
            else:
                dots = [];
                for annotation_id in range(len(self.annos[idx]['annotation'])):
                    annotation_item = self.annos[idx]['annotation'][annotation_id]
                    dot = [annotation_item['x'], annotation_item['y']]
                    x, y = dot[0], dot[1]
                    if x > 0 and x < wd  and y > 0 and y < ht:
                    # dot_t = [roi[0] + roi[2]/2., roi[1] + roi[2] / 4.] 
                        dots.append(dot)                
                if len(dots) == 0:
                    print "-------error---------"
                self.points_list.append(dots)

        # if self.pre_load:
        #     for idx in range(self.num_samples):
        #         print('pre load :', idx)
        #         img_name = self.base_image_path + self.annos[idx]['name']
        #         img = cv2.imread(img_name, 0)
        #         img = img.astype(np.float32, copy=False)

        #         if self.img_normalization_method == 'SubMean': #  'ImageNet'
        #             img = img - 127
        #         elif self.img_normalization_method == 'ImageNet':
        #             img = (img/255. - 0.45) / 0.225

        #                              #   transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                              # std=[0.229, 0.224, 0.225]),
        #         self.images_list.append(img)
        #         den_name = '%s/%06d.csv'%(self.density_gt_path, idx)
        #         den = pd.read_csv(den_name, sep=',',header=None).as_matrix()     
        #         den = den.astype(np.float32, copy=False)
        #         self.dens_list.append(den)


    def __iter__(self):
        # if self.shuffle:            
        #     random.shuffle(self.annos)

        # current_annos = self.annos
        # id_list = self.id_list
        # self.Blob
        # self.batch_size
        if self.training:
            train_ids_temp = self.train_ids
            np.random.shuffle(train_ids_temp)

            iters = int(np.floor(self.num_train_images / self.batch_size))

            for iter_id in range(iters):
                blob = {}
                img_list = []
                den_1_list = []
                den_2_list = []
                gt_nums = []
                wrong_tag = False
                for batch_id in range(self.batch_size):
                    
                    idx = iter_id * self.batch_size + batch_id
                    current_img_id = train_ids_temp[idx]
                    img_idx = current_img_id
                    # img_idx = self.annos_origin[current_img_id]['id']
                    # if not(img_idx == current_img_id):
                    #     print('Wrong error!!!')
                    # print('mean ', np.sum(img) / (img.shape[0]*img.shape[1]))
                    # img = img - 125.;
                    if self.pre_load:
                        img = self.images_list[img_idx]
                        den = self.dens_list[img_idx]
                    else:
                        # img_name = self.base_image_path + current_annos[idx]['name']
                        # img = cv2.imread(img_name, 0)
                        # img = img.astype(np.float32, copy=False)


                        ######################### 
                        #  paddle image loading
                        ######################### 
                        # img_name = self.base_image_path + self.annos_origin[idx]['name']
                        # img = paddle.image.load_image(img_name, is_color=False)
                        # img = cv2.resize(img, (1024, 512), interpolation=cv2.INTER_CUBIC)

                        # print(type(img))
                        ######################### 
                        # load image files from npy files
                        ########################## 
                        # npy file 
                        wrong_id_list = [54, 211, 222, 517, 1070, 1983, 2015, 2237, 2625, 2852]
                        id_list = [3625+i for i in wrong_id_list]
                        if img_idx in id_list:
                            print(img_idx)
                            wrong_tag = True
                            continue

                        image_npy_name = self.img_npy_path + '%06d.npy'%(img_idx)
                        img = np.load(image_npy_name)

                        ## ignore region processing
                        if len(self.annos[img_idx]['ignore_region']):
                            #print("ignore_region_num", len(self.annos[img_idx]['ignore_region'])-1)
                            for ignore_region_id in range(len(self.annos[img_idx]['ignore_region'])):
                                dots_IR = []
                                for annotation_id in range(len(self.annos[img_idx]['ignore_region'][ignore_region_id])):
                                    annotation_item = self.annos[img_idx]['ignore_region'][ignore_region_id][annotation_id]
                                    dot_IR = [annotation_item['x'], annotation_item['y']]
                                    
                                    dots_IR.append(dot_IR)
                            
                                cv2.fillPoly(img, [np.array(dots_IR)], 127)

                        if self.img_normalization_method == 'SubMean': #  'ImageNet'
                            img = img - 127
                        elif self.img_normalization_method == 'ImageNet':
                            img = (img/255. - 0.45) / 0.225

                        #_vis_minibatch(img)
                        # img = img - 127.;
                        # img = (img/255. - 0.45) / 0.225;
                        # den_name = '%s/%06d.csv'%(self.density_gt_path, img_idx)
                        den_npy_name = self.den_npy_path + '%06d.npy'%(img_idx)
                        den = np.load(den_npy_name)

                        # den = pd.read_csv(den_name, sep=',',header=None).as_matrix()  
                        # den = den.astype(np.float32, copy=False)

                    #print('img.shape: ', img.shape)
                    ht = img.shape[0]
                    wd = img.shape[1]

                    # print('sum ', np.sum(den))
                    # print('den.shape: ', den.shape)
                    crop_h = int(np.floor(ht / self.patch_size_ratio)) # 
                    crop_w = int(np.floor(wd / self.patch_size_ratio))   

                    crop_h = (crop_h/4)*4
                    crop_w = (crop_w/4)*4
                    # print('crop_h: ', crop_h, ' ', crop_w)
                    points = self.points_list[img_idx]   
                    points_num = len(points)


                    if points_num > self.sparse_thre: # large number
                        random_crop = True
                    else:
                        if np.random.rand(1) > 0.5:
                            random_crop = True
                        else:
                            random_crop = False

                    # random_crop = True
                    if random_crop:
                        for crop_id in range(self.samples_to_crop_large):   
                            den_crop = np.zeros((crop_h, crop_w), dtype=np.float32)
                            id = 0
                            while np.sum(den_crop) == 0:
                                id +=1 
                                crop_start_x = int(np.floor(np.random.rand(1) * (wd - crop_w)))
                                crop_start_y = int(np.floor(np.random.rand(1) * (ht - crop_h)))
                                im_crop = img[crop_start_y:crop_start_y + crop_h, crop_start_x:crop_start_x + crop_w,:]
                                den_crop = den[crop_start_y:crop_start_y + crop_h, crop_start_x:crop_start_x + crop_w]
                                # print('im_crop.shape: ', im_crop.shape)
                                if self.gt_downsample:
                                    wd_1 = crop_w/4
                                    ht_1 = crop_h/4
                                    den_1_crop = cv2.resize(den_crop,(wd_1,ht_1)) 
                                    den_1_crop = den_1_crop * ((crop_w*crop_h)/(wd_1*ht_1))
                                    den_2_crop = den_crop

                            #data augmentation on the fly
                            if np.random.uniform() > 0.5:
                                #randomly flip input image and density 
                                im_crop = np.flip(im_crop,1).copy()
                                den_1_crop = np.flip(den_1_crop,1).copy()
                                den_2_crop = np.flip(den_2_crop,1).copy()
                            
                            # if np.random.uniform() > 0.5:
                            #     #add random noise to the input image
                            #     im_crop = im_crop + np.random.uniform(-10,10,size=im_crop.shape) 
        
                            img_list.append(im_crop)
                            den_1_list.append(den_1_crop)
                            den_2_list.append(den_2_crop)
                            gt_num = self.annos_origin[current_img_id]['num'] 
                            gt_nums.append(gt_num)
                            #_vis_minibatch(im_crop)
                            #_vis_minibatch(den_1_crop)
                            #_vis_minibatch(den_2_crop)
                    else: # small number
                        p_idx_list = [idx for idx in range(points_num)]
                        random.shuffle(p_idx_list)
                        # num_points = len(p_idx_list)
                        for crop_id in range(self.samples_to_crop_small):
                            den_crop = np.zeros((crop_h, crop_w), dtype=np.float)
                            while np.sum(den_crop)==0:
                                # if crop_id >= points_num:
                                id_involved = crop_id % points_num
                                # else:
                                    # id_involved = crop_id
                                p_idx = p_idx_list[id_involved]
                                point = points[p_idx]
                                x, y = point[0], point[1]
                                # while(1):
                                crop_start_x = x - crop_w + (np.random.rand(1) * crop_w)
                                crop_start_y = y - crop_h + (np.random.rand(1) * crop_h)
                                crop_end_x = crop_start_x + crop_w
                                crop_end_y = crop_start_y + crop_h
                                crop_start_x, crop_start_y, crop_end_x, crop_end_y = \
                                 _clip_start_end(wd, ht, crop_start_x, crop_start_y, crop_end_x, crop_end_y)

                                crop_w = crop_end_x - crop_start_x
                                crop_h = crop_end_y - crop_start_y

                                im_crop = img[crop_start_y:crop_start_y + crop_h, crop_start_x:crop_start_x + crop_w,:]
                                den_crop = den[crop_start_y:crop_start_y + crop_h, crop_start_x:crop_start_x + crop_w]
                                # if crop_w <= 8 or crop_h <= 4:
                                #     continue
                                # # print('im_crop.shape: ', im_crop.shape)
                                if self.gt_downsample:
                                    wd_1 = crop_w/4
                                    ht_1 = crop_h/4
                                    den_1_crop = cv2.resize(den_crop,(wd_1,ht_1))                
                                    den_1_crop = den_1_crop * ((crop_w*crop_h)/(wd_1*ht_1))
                                    den_2_crop = den_crop

                            #data augmentation on the fly
                            if np.random.uniform() > 0.5:
                                #randomly flip input image and density 
                                im_crop = np.flip(im_crop,1).copy()
                                den_1_crop = np.flip(den_1_crop,1).copy()
                                den_2_crop = np.flip(den_2_crop,1).copy()
                            # if np.random.uniform() > 0.5:
                            #     #add random noise to the input image
                            #     im_crop = im_crop + np.random.uniform(-10,10,size=im_crop.shape) 

                            # print('np.sum(den_crop: ', np.sum(den_crop))
                            img_list.append(im_crop)
                            den_1_list.append(den_1_crop)
                            den_2_list.append(den_2_crop)   
                            gt_num = self.annos_origin[current_img_id]['num'] 
                            gt_nums.append(gt_num)
                            # print(im_crop.shape)
                            #_vis_minibatch(im_crop)
                            #_vis_minibatch(den_1_crop)
                            #_vis_minibatch(den_2_crop)
                            # print(im_crop.shape)
                            # print(den_crop.shape)
                # print(' ')
                # blob['data']=_im_list_to_blob(img_list)

                # blob['gt_density']=_den_list_to_blob(den_list)
                # blob['gt_nums'] = gt_nums
                # image.flatten()
                if wrong_tag:
                    continue

                blob['data']=_im_list_to_blob(img_list)#.flatten()
                blob['gt_1_density']=_den_list_to_blob(den_1_list)#.flatten()
                blob['gt_2_density']=_den_list_to_blob(den_2_list)
                blob['gt_nums'] = gt_nums
                # blob['fname'] = fname

                yield blob

        elif self.testing:

            iters= int(np.floor(self.num_test_images / self.test_batch_size))

            for iter_id in range(iters):
                blob = {}
                img_list = []
                den_list = []
                gt_nums = []

                for batch_id in range(self.test_batch_size):
                    idx = iter_id * self.test_batch_size + batch_id
                    image_name = self.base_image_path_test + self.annos_test[idx]['name']
                    current_test_img = cv2.imread(image_name)
                    shape_o = current_test_img.shape
                    h, w = shape_o[0], shape_o[1]
                    ratio_h = h / 576.
                    ratio_w = w / 1024.
                    current_test_img = cv2.resize(current_test_img, (1024, 576))
                    #current_test_img = current_test_img.reshape((1024, 576, 3))

                    ## ignore region processing
                    if len(self.annos_test[idx]['ignore_region']):
                        print("ignore_region_num", len(self.annos_test[idx]['ignore_region']))
                        for ignore_region_id in range(len(self.annos_test[idx]['ignore_region'])):
                            dots_IR = []
                            for annotation_id in range(len(self.annos_test[idx]['ignore_region'][ignore_region_id])):
                                annotation_item = self.annos_test[idx]['ignore_region'][ignore_region_id][annotation_id]
                                dot_IR = [int(annotation_item['x']/ratio_w), int(annotation_item['y']/ratio_h)]
                                dots_IR.append(dot_IR)
                                
                            cv2.fillPoly(current_test_img, [np.array(dots_IR)], 127)
                        print("ignore ready--------")
                    # current_test_img = current_test_img[:,:,[2,1,0]]
                    # _vis_minibatch(current_test_img)

                    if self.img_normalization_method == 'SubMean': #  'ImageNet'
                        current_test_img = current_test_img - 127
                    elif self.img_normalization_method == 'ImageNet':
                        current_test_img = (current_test_img/255. - 0.45) / 0.225                    
                    # current_test_img = current_test_img - 127.
                    
                    img_list.append(current_test_img)
                    # .flatten()
                blob['data'] = _im_list_to_blob(img_list)
                # blob['data'] = img_list[0]#.flatten()

                yield blob

        else:
            val_ids_temp = self.val_ids
         
            iters = int(np.floor(self.num_val_images / self.test_batch_size))
            
            for iter_id in range(iters):
                blob = {}
                img_list = []
                den_1_list = []
                den_2_list= []
                gt_nums = []
                wrong_tag = False
                for batch_id in range(self.test_batch_size):
                    
                    idx = iter_id * self.test_batch_size + batch_id
                    current_img_id = val_ids_temp[idx]
                    img_idx = current_img_id
                    # img_idx = self.annos_origin[current_img_id]['id']
                    # if not(img_idx == current_img_id):
                    #     print('Wrong error!!!')
                    # print('mean ', np.sum(img) / (img.shape[0]*img.shape[1]))
                    # img = img - 125.;
                    if self.pre_load:
                        img = self.images_list[img_idx]
                        den = self.dens_list[img_idx]
                    else:
                        # img_name = self.base_image_path + current_annos[idx]['name']
                        # img = cv2.imread(img_name, 0)
                        # img = img.astype(np.float32, copy=False)
                        wrong_id_list = [54, 211, 222, 517, 1070, 1983, 2015, 2237, 2625, 2852]
                        id_list = [3625+i for i in wrong_id_list]
                        if img_idx in id_list:
                            wrong_tag = True
                            print img_idx
                            continue

                        image_npy_name = self.img_npy_path + '%06d.npy'%(img_idx)
                        img = np.load(image_npy_name)

                        ## ignore region processing
                        if len(self.annos[img_idx]['ignore_region']):
                            for ignore_region_id in range(len(self.annos[img_idx]['ignore_region'])):
                                dots_IR = []
                                for annotation_id in range(len(self.annos[img_idx]['ignore_region'][ignore_region_id])):
                                    annotation_item = self.annos[img_idx]['ignore_region'][ignore_region_id][annotation_id]
                                    dot_IR = [annotation_item['x'], annotation_item['y']]
                                    dots_IR.append(dot_IR)
                                    
                                cv2.fillPoly(img, [np.array(dots_IR)], 127)

                        if self.img_normalization_method == 'SubMean': #  'ImageNet'
                            img = img - 127
                        elif self.img_normalization_method == 'ImageNet':
                            img = (img/255. - 0.45) / 0.225
                        # img = img - 127.;
                        # img = (img/255. - 0.45) / 0.225;
                        # den_name = '%s/%06d.csv'%(self.density_gt_path, img_idx)
                        den_npy_name = self.den_npy_path + '%06d.npy'%(img_idx)
                        den = np.load(den_npy_name)

                    # print('img.shape: ', img.shape)
                    ht = img.shape[0]
                    wd = img.shape[1]

                    ht1 = (ht/4) * 4
                    wd1 = (wd/4) * 4
                    #img = cv2.resize(img,(wd1,ht1,3))    

                    # print('im_crop.shape: ', im_crop.shape)
                    if self.gt_downsample:
                        wd_1 = wd1/4
                        ht_1 = ht1/4
                        den_1 = cv2.resize(den,(wd_1,ht_1))                
                        den_1 = den_1 * ((ht*wd)/(wd_1*ht_1))
                        den_2 = den
                    # print('np.sum(den_crop: ', np.sum(den_crop))
                    img_list.append(img)
                    den_1_list.append(den_1)
                    den_2_list.append(den_2)   
                    gt_num = self.annos_origin[current_img_id]['num'] 
                    gt_nums.append(gt_num)     

                    #_vis_minibatch(img)
                    # _vis_minibatch(den)
                if wrong_tag:
                    continue
                # blob['data']=img_list[0]#.flatten()
                # blob['gt_density']=den_list[0]#.flatten()
                # blob['gt_nums'] = gt_nums

                blob['data']=_im_list_to_blob(img_list)

                blob['gt_1_density']=_den_list_to_blob(den_1_list)
                blob['gt_2_density']=_den_list_to_blob(den_2_list)
                blob['gt_nums'] = gt_nums
                # blob['gt_num']
                # blob['fname'] = fname
                yield blob            


    def get_num_samples(self):
        if self.training:
            return self.num_train_images    
        else:
            return self.num_val_images
                
        
    # def _vis_minibatch(img):
    #     """Visualize a mini-batch for debugging."""
    #     import matplotlib.pyplot as plt
    #     plt.imshow(img)
    #     plt.show()
        # for i in xrange(rois_blob.shape[0]):
        #     rois = rois_blob[i, :]
        #     im_ind = rois[0]
        #     roi = rois[1:]
        #     im = im_blob[im_ind, :, :, :].transpose((1, 2, 0)).copy()
        #     im += cfg.PIXEL_MEANS
        #     im = im[:, :, (2, 1, 0)]
        #     im = im.astype(np.uint8)
        #     cls = labels_blob[i]
        #     plt.imshow(im)
        #     print 'class: ', cls, ' overlap: ', overlaps[i]
        #     plt.gca().add_patch(
        #         plt.Rectangle((roi[0], roi[1]), roi[2] - roi[0],
        #                       roi[3] - roi[1], fill=False,
        #                       edgecolor='r', linewidth=3)
        #         )
           
           
        
