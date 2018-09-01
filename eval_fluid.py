# -*- coding:utf-8 -*-
from __future__ import print_function
import time
import numpy as np
import os
import cv2
try:
    from termcolor import cprint
except ImportError:
    cprint = None

# import paddle
import paddle.fluid as fluid
import numpy
from src.data_loader_BDS_fluid_Stage2 import ImageDataLoader # src.data_loader :orginal version  data_loader : pyramid version
#from src.fluid_model import *
from src.fluid_bds_model import *
import matplotlib.pyplot as plt

# import numpy as np
# import paddle.fluid as fluid
import paddle.v2 as paddle
try:
    from termcolor import cprint
except ImportError:
    cprint = None

# if not os.path.exists(output_dir):
#     os.mkdir(output_dir)

def _vis_minibatch(img):
    """Visualize a mini-batch for debugging."""
    img = np.transpose(img[0], (1,2,0))
    plt.imshow(img)
    plt.show()

def _vis_denmap(den):
    a = np.squeeze(den[0], axis=[0,1])
    # print((a.shape))
    plt.imshow(a)
    plt.show()
    cv2.waitKey(1000)

def log_print(text, color=None, on_color=None, attrs=None):
    if cprint is not None:
        cprint(text, color=color, on_color=on_color, attrs=attrs)
    else:
        print(text)

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def mkdirs(path):
	if not os.path.exists(path):
	    os.mkdir(path)

BATCH_SIZE = 1
#root_path = '/data0/zsq/Research_2/CrowdCounting/dataset/BDS/'
root_path = '/data1/ShengGe/ZSQ/crowdcount-mcnn/'

data_loader_val=ImageDataLoader(root_path, training = False, batch_size = 1, shuffle=True, gt_downsample=True, pre_load=False)


# def val_data():
#     def reader():
#         for blob in data_loader_val: # image num
#             tmp_xs = blob["data"]
#             tmp_ys = blob["gt_density"]
#             # _vis_minibatch(tmp_xs)
#             # _vis_minibatch(tmp_ys)
#             yield tmp_xs, tmp_ys  # size    ((1,1,img.shape[0],img.shape[1])), (1,1,den.shape[0],den.shape[1])
#     return reader()

image_val_shape = [3, 576, 1024]
den_val_1_shape = [1, 144, 256]
den_val_2_shape = [1, 576, 1024]

image_val = fluid.layers.data(name='image_val', shape=image_val_shape, dtype='float32')
# 定义标签，类型是整型
den_val_1 = fluid.layers.data(name='den_val_1', shape=den_val_1_shape, dtype='float32')
den_val_2 = fluid.layers.data(name='den_val_2', shape=den_val_2_shape, dtype='float32')

#MCNN = mcnn_program(image_val)
#MCNN =inference_bn(image_val)
MCNN = iterative_crowd_counting(image_val)
out_put_1, out_put_2= MCNN.top_layer
print(out_put_1.shape, out_put_2.shape)
# loss = fluid.layers.square_error_cost(out_put, den)
# avg_loss = fluid.layers.mean(loss)

# optimizer = fluid.optimizer.Adam(learning_rate=1e-5)
# optimizer.minimize(avg_loss)

#model_name = '_iterative_crowd_counting_dot_bbox'
model_name = '_iterative_crowd_counting_dot_bbox_twostage'


output_dir = './saved_models/fluid' + model_name
mkdirs(output_dir)
train_check = output_dir + '/train/'
mkdirs(train_check)
test_check = output_dir + '/test/'
mkdirs(test_check)

place = fluid.CUDAPlace(0) # gpu id
exe = fluid.Executor(place)

exe.run(fluid.default_startup_program())

# val_reader = paddle.batch(
#         paddle.reader.shuffle(val_data, buf_size=5000),
#         batch_size=BATCH_SIZE)

# feeder = fluid.DataFeeder(place=place, feed_list=[image, den])
# feader_val = fluid.DataFeeder(place=place, feed_list=[image_val, den_val])

#best_mae_BDS = 0.6483
#best_mae_BDS_epoch = 6
#best_mae = 15.9089
#best_mae_epoch = 4

best_mae_BDS =1e6
best_mae_BDS_epoch = -1
best_mae = 1e6
best_mae_epoch = -1

# EPOCH: 122, MAE: 20.0952, MSE: 88.4072 MAEBDS: 0.4258
# BEST_MAE_EPOCH: 109, BEST_MAE: 12.3038
# BEST_MAEBDS_EPOCH: 105, BEST_MAEBDS: 0.3393   
### use ignore region

for epoch in range(0, 1000):
    param_path = test_check + 'epoch_%06d'%(epoch)

    # block
    while(not os.path.exists(param_path)):
        time.sleep(3)
        param_path = test_check + 'epoch_%06d'%(epoch)
    time.sleep(3)

    # process
    prog = fluid.default_main_program()
    print('loading...%s'%(param_path))
    fluid.io.load_params(executor=exe, dirname=param_path,
                         main_program=prog)
    mae = 0.0
    mae_BDS = 0.0
    mse = 0.0
    idx = 0
    batch_id = 0
    # for batch_id, data in enumerate(val_reader()):
    #   if batch_id % 100 == 0:
    #       print('epoch: %05d  batch_id: %05d'%(pass_id, batch_id))
    # # print('starting..epoch' , pass_id,  ' b',batch_id)
    #     den_es = exe.run(
    #         fluid.framework.default_main_program(),
    #         feed=feader_val.feed(data),
    #         fetch_list=[out_put])

    for blob in data_loader_val:
        if batch_id % 100 == 0:
            print('epoch: %05d  batch_id: %05d'%(epoch, batch_id))        
        val_img = blob['data']
        #print(val_img.shape)
        den_es_1, den_es_2 = exe.run(
            prog,
            feed={'image_val': val_img},
            fetch_list=[out_put_1, out_put_2])  

        #_vis_minibatch(val_img)
        #_vis_denmap(den_es_2)

        et_count_1 = np.sum(den_es_1)
        et_count_2 = np.sum(den_es_2)
        gt_count = blob['gt_nums'][0]
        mae += abs(gt_count-et_count_2)
        mae_BDS += float(abs(gt_count-et_count_2))/(gt_count)
        mse += ((gt_count-et_count_2)*(gt_count-et_count_2))    
        batch_id +=1  
        # idx += 1
    mae = mae/float(data_loader_val.get_num_samples())
    mae_BDS = mae_BDS / float(data_loader_val.get_num_samples())
    mse = np.sqrt(mse/float(data_loader_val.get_num_samples()))

    if mae_BDS < best_mae_BDS:
        best_mae_BDS = mae_BDS
        best_mae_BDS_epoch = epoch
    if mae < best_mae:
        best_mae = mae
        best_mae_epoch = epoch

    log_text = 'EPOCH: %d, MAE: %.4f, MSE: %0.4f MAEBDS: %.4f' % (epoch,mae,mse,mae_BDS)
    log_print(log_text, color='green', attrs=['bold'])

    log_text = 'BEST_MAE_EPOCH: %d, BEST_MAE: %.4f' % (best_mae_epoch,best_mae)
    log_print(log_text, color='green', attrs=['bold'])

    log_text = 'BEST_MAEBDS_EPOCH: %d, BEST_MAEBDS: %.4f' % (best_mae_BDS_epoch,best_mae_BDS)
    log_print(log_text, color='green', attrs=['bold'])


    # print('loss: %.7f', loss_data)
    # # training
    # print('saving.. epoch_%06d'%(pass_id))
    # path = train_check + 'epoch_%06d'%(pass_id)
    # mkdirs(path)
    # prog = fluid.default_main_program()
    # trainer_args = {"epoch_id": pass_id,"step_id": 100}
    # fluid.io.save_checkpoint(executor=exe,
    #                         checkpoint_dir=path,
    #                         trainer_id=0,
    #                         trainer_args=trainer_args,
    #                         main_program=prog,
    #                         max_num_checkpoints=1)
    #test 
    # param_path = test_check + 'epoch_%06d'%(pass_id)
    # mkdirs(param_path)
    # prog = fluid.default_main_program()
    # fluid.io.save_params(executor=exe, dirname=param_path, main_program=prog)


# exe = fluid.Executor(fluid.CPUPlace())
# param_path = "./my_paddle_model"

        # print("Cur Cost : %f" % (np.array(loss[0])[0]))


# import numpy as np

# import paddle.fluid as fluid
# import paddle.v2 as paddle

# define the input layers for the network.
# x = fluid.layers.data(name="img", shape=[1, 28, 28], dtype="float32")
# y_ = fluid.layers.data(name="label", shape=[1], dtype="int64")


# BATCH_SIZE = 100

# y = fluid.layers.fc(input=x, size=10, act="softmax")
# loss = fluid.layers.cross_entropy(input=y, label=y_)
# avg_loss = fluid.layers.mean(loss)

# # define the optimization algorithm.
# optimizer = fluid.optimizer.Adam(learning_rate=1e-3)
# optimizer.minimize(avg_loss)



# place = fluid.CPUPlace()
# exe = fluid.Executor(place)
# exe.run(fluid.default_startup_program())

# train_reader = paddle.batch(
#         paddle.reader.shuffle(paddle.dataset.mnist.train(), buf_size=5000),
#         batch_size=BATCH_SIZE)
# feeder = fluid.DataFeeder(place=place, feed_list=[x, y_])

# for pass_id in range(100):
#     for batch_id, data in enumerate(train_reader()):
#         loss = exe.run(
#             fluid.framework.default_main_program(),
#             feed=feeder.feed(data),
#             fetch_list=[avg_loss])
#         print("Cur Cost : %f" % (np.array(loss[0])[0]))
# batch_size = fluid.layers.create_tensor(dtype='int64')
# print batch_size
# # batch_acc = fluid.layers.accuracy(input=predict, label=label, total=batch_size)

# inference_program = fluid.default_main_program().clone(for_test=True)




# optimizer = fluid.optimizer.Momentum(
#     learning_rate=fluid.layers.exponential_decay(
#         learning_rate=learning_rate,
#         decay_steps=40000,
#         decay_rate=0.1,
#         staircase=True),
#     momentum=0.9,
#     regularization=fluid.regularizer.L2Decay(0.0005), )
# opts = optimizer.minimize(loss)
# use_cuda = True
# place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
# # 创建调试器
# exe = fluid.Executor(place)
# # 初始化调试器
# exe.run(fluid.default_startup_program())


