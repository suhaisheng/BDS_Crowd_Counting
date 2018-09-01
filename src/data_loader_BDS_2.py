import numpy as np
import cv2
import os
import random
import json
import pandas as pd
# import cv2
import paddle.v2 as paddle
# import matplotlib.pyplot as plt

def _im_list_to_blob(ims):
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
        end_y = 256
    # if end_x >= w:
    #     end_x = w
        # start_x = end_x - crop_w_tmp
    if end_x >= 1024:
        end_x = 1024
        start_x = 512
    # if end_y >= h:
    #     end_y = h
    #     start_y = end_y - crop_h_tmp
    if end_y >= 512:
        end_y = 512
        start_y = 256
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
        self.f = open(self.root_path + "/images/baidu_star_2018_train_stage1/baidu_star_2018/annotation/new_anno_xy_transfered_stage1.json")
        self.annotations = json.load(self.f)
        self.f.close()

        self.f_test = open(self.root_path + "/images/baidu_star_2018_test_stage1/baidu_star_2018/annotation/annotation_test_stage1.json")
        self.annotations_test = json.load(self.f_test)
        self.f_test.close()

        self.base_image_path = self.root_path + "/images/baidu_star_2018_train_stage1/baidu_star_2018/image/"
        self.base_image_path_test = self.root_path + "/images/baidu_star_2018_test_stage1/baidu_star_2018/image/"
        # self.
        self.training = training
        self.testing = testing 

        self.img_normalization_method = 'SubMean' # 'ImageNet' # 

        self.density_gt_path = self.root_path + "/images/baidu_star_2018_train_stage1/baidu_star_2018/GT_Density/"
        # images are gray images
        # density maps are 1 channel
        self.img_npy_path = self.root_path + "/images/baidu_star_2018_train_stage1/baidu_star_2018/image_1_resize_npy/"
        self.den_npy_path = self.root_path + "/images/baidu_star_2018_train_stage1/baidu_star_2018/dmap_resize/"

        self.annos_test = self.annotations_test['annotations']
        self.num_test_images = len(self.annos_test)

        self.annos = self.annotations['annotations']
        self.annos_origin = self.annotations['annotations']
        # self.annos_train = 
        # self.annos_val = 

        # self.train_ids = np.load("/home/fanglj/ZSQ/Research_2/CrowdCounting/crowdcount-mcnn/images/baidu_star_2018_train_stage1/baidu_star_2018/train_ids.npy")
        # self.val_ids = np.load("/home/fanglj/ZSQ/Research_2/CrowdCounting/crowdcount-mcnn/images/baidu_star_2018_train_stage1/baidu_star_2018/val_ids.npy")
        self.train_ids = np.load(self.root_path + "/images/baidu_star_2018_train_stage1/baidu_star_2018/train_ids_41.npy").astype(np.int64)
        self.val_ids = np.load(self.root_path + "/images/baidu_star_2018_train_stage1/baidu_star_2018/val_ids_41.npy").astype(np.int64)

        self.num_images = len(self.annos)

        self.num_train_images = self.train_ids.shape[0]
        self.num_val_images = self.val_ids.shape[0]

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

        for idx in range(self.num_samples):
            anno_type = self.annos[idx]['type']
            # img_name = self.base_image_path + self.annos[idx]['name']
            ht = 512
            wd = 1024

            if anno_type == 'bbox':
                dots = [];
                print('error')
                for annotation_id in range(len(self.annos[idx]['annotation'])):
                    annotation_item = self.annos[idx]['annotation'][annotation_id]
                    roi = [annotation_item['x'], annotation_item['y'], annotation_item['w'], annotation_item['h']]
                    dot_t = [roi[0] + roi[2]/2., roi[1] + roi[2] / 4.] 
                    x, y = dot_t[0], dot_t[1]
                    if x > 0 and x < wd  and y > 0 and y < ht:
                        dots.append(dot_t)   
                    
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
                
                self.points_list.append(dots)

        if self.pre_load:
            for idx in range(self.num_samples):
                print('pre load :', idx)
                img_name = self.base_image_path + self.annos[idx]['name']
                img = cv2.imread(img_name, 0)
                img = img.astype(np.float32, copy=False)

                if self.img_normalization_method == 'SubMean': #  'ImageNet'
                    img = img - 127
                elif self.img_normalization_method == 'ImageNet':
                    img = (img/255. - 0.45) / 0.225

                                     #   transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     # std=[0.229, 0.224, 0.225]),
                self.images_list.append(img)
                den_name = '%s/%06d.csv'%(self.density_gt_path, idx)
                den = pd.read_csv(den_name, sep=',',header=None).as_matrix()     
                den = den.astype(np.float32, copy=False)
                self.dens_list.append(den)


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
                den_list = []
                gt_nums = []
                for batch_id in range(self.batch_size):
                    
                    idx = iter_id * self.batch_size + batch_id
                    current_img_id = train_ids_temp[idx]

                    img_idx = self.annos_origin[current_img_id]['id']
                    if not(img_idx == current_img_id):
                        print('Wrong error!!!')
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
                        image_npy_name = self.img_npy_path + '%06d.npy'%(img_idx)
                        img = np.load(image_npy_name)


                        if self.img_normalization_method == 'SubMean': #  'ImageNet'
                            img = img - 127
                        elif self.img_normalization_method == 'ImageNet':
                            img = (img/255. - 0.45) / 0.225
                        # img = img - 127.;
                        # img = (img/255. - 0.45) / 0.225;
                        # den_name = '%s/%06d.csv'%(self.density_gt_path, img_idx)
                        den_npy_name = self.den_npy_path + '%06d.npy'%(img_idx)
                        den = np.load(den_npy_name)

                        # den = pd.read_csv(den_name, sep=',',header=None).as_matrix()  
                        # den = den.astype(np.float32, copy=False)

                    # print('img.shape: ', img.shape)
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
                        if np.random.rand(1) > 0.7:
                            random_crop = True
                        else:
                            random_crop = False

                    # random_crop = True
                    if random_crop:
                        for crop_id in range(self.samples_to_crop_large):       
                            crop_start_x = int(np.floor(np.random.rand(1) * (wd - crop_w)))
                            crop_start_y = int(np.floor(np.random.rand(1) * (ht - crop_h)))
                            im_crop = img[crop_start_y:crop_start_y + crop_h, crop_start_x:crop_start_x + crop_w]
                            den_crop = den[crop_start_y:crop_start_y + crop_h, crop_start_x:crop_start_x + crop_w]
                            # print('im_crop.shape: ', im_crop.shape)
                            if self.gt_downsample:
                                wd_1 = crop_w/4
                                ht_1 = crop_h/4
                                den_crop = cv2.resize(den_crop,(wd_1,ht_1))                
                                den_crop = den_crop * ((crop_w*crop_h)/(wd_1*ht_1))
                            # print('np.sum(den_crop: ', np.sum(den_crop))
                            img_list.append(im_crop)
                            den_list.append(den_crop)
                            gt_num = self.annos_origin[current_img_id]['num'] 
                            gt_nums.append(gt_num)
                            # _vis_minibatch(im_crop)
                            # _vis_minibatch(den_crop)
                    else: # small number
                        p_idx_list = [idx for idx in range(points_num)]
                        random.shuffle(p_idx_list)
                        # num_points = len(p_idx_list)
                        for crop_id in range(self.samples_to_crop_small):

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



                            im_crop = img[crop_start_y:crop_start_y + crop_h, crop_start_x:crop_start_x + crop_w]
                            den_crop = den[crop_start_y:crop_start_y + crop_h, crop_start_x:crop_start_x + crop_w]
                            # if crop_w <= 8 or crop_h <= 4:
                            #     continue
                            # # print('im_crop.shape: ', im_crop.shape)
                            if self.gt_downsample:
                                wd_1 = crop_w/4
                                ht_1 = crop_h/4
                                den_crop = cv2.resize(den_crop,(wd_1,ht_1))                
                                den_crop = den_crop * ((crop_w*crop_h)/(wd_1*ht_1))
                            # print('np.sum(den_crop: ', np.sum(den_crop))
                            img_list.append(im_crop)
                            den_list.append(den_crop)   
                            gt_num = self.annos_origin[current_img_id]['num'] 
                            gt_nums.append(gt_num)
                            # print(im_crop.shape)
                            # _vis_minibatch(im_crop)
                            # _vis_minibatch(den_crop)
                            # print(im_crop.shape)
                            # print(den_crop.shape)
                # print(' ')
                # blob['data']=_im_list_to_blob(img_list)

                # blob['gt_density']=_den_list_to_blob(den_list)
                # blob['gt_nums'] = gt_nums
                # image.flatten()

                blob['data']=img_list[0]#.flatten()

                blob['gt_density']=den_list[0]#.flatten()

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
                    current_test_img = cv2.imread(image_name, 0)
                    current_test_img = cv2.resize(current_test_img, (1024, 512))

                    if self.img_normalization_method == 'SubMean': #  'ImageNet'
                        current_test_img = current_test_img - 127
                    elif self.img_normalization_method == 'ImageNet':
                        current_test_img = (current_test_img/255. - 0.45) / 0.225                    
                    # current_test_img = current_test_img - 127.

                    img_list.append(current_test_img)
                    # .flatten()
                # blob['data'] = _im_list_to_blob(img_list)
                blob['data'] = img_list[0]#.flatten()

                yield blob

        else:
            val_ids_temp = self.val_ids
            iters = int(np.floor(self.num_val_images / self.test_batch_size))

            for iter_id in range(iters):
                blob = {}
                img_list = []
                den_list = []
                gt_nums = []

                for batch_id in range(self.test_batch_size):
                    
                    idx = iter_id * self.test_batch_size + batch_id
                    current_img_id = val_ids_temp[idx]

                    img_idx = self.annos_origin[current_img_id]['id']
                    if not(img_idx == current_img_id):
                        print('Wrong error!!!')
                    # print('mean ', np.sum(img) / (img.shape[0]*img.shape[1]))
                    # img = img - 125.;
                    if self.pre_load:
                        img = self.images_list[img_idx]
                        den = self.dens_list[img_idx]
                    else:
                        # img_name = self.base_image_path + current_annos[idx]['name']
                        # img = cv2.imread(img_name, 0)
                        # img = img.astype(np.float32, copy=False)

                        image_npy_name = self.img_npy_path + '%06d.npy'%(img_idx)
                        img = np.load(image_npy_name)

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
                    img = cv2.resize(img,(wd1,ht1))    

                    # print('im_crop.shape: ', im_crop.shape)
                    if self.gt_downsample:
                        wd_1 = wd1/4
                        ht_1 = ht1/4
                        den = cv2.resize(den,(wd_1,ht_1))                
                        den = den * ((ht*wd)/(wd_1*ht_1))
                    # print('np.sum(den_crop: ', np.sum(den_crop))
                    img_list.append(img)
                    den_list.append(den)   
                    gt_num = self.annos_origin[current_img_id]['num'] 
                    gt_nums.append(gt_num)     

                    # _vis_minibatch(img)
                    # _vis_minibatch(den)

                blob['data']=img_list[0]#.flatten()

                blob['gt_density']=den_list[0]#.flatten()

                blob['gt_nums'] = gt_nums

                # blob['data']=_im_list_to_blob(img_list)

                # blob['gt_density']=_den_list_to_blob(den_list)

                # blob['gt_nums'] = gt_nums
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
           
           
        
