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
from src.data_loader_BDS_fluid_Stage2 import ImageDataLoader   # src.data_loader :orginal version  data_loader : pyramid version
#from src.fluid_model import *
from src.fluid_mscnn_model import *

# import numpy as np
# import paddle.fluid as fluid
import paddle.v2 as paddle
try:
    from termcolor import cprint
except ImportError:
    cprint = None

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
root_path = '/data1/ShengGe/ZSQ/crowdcount-mcnn'

data_loader_train = ImageDataLoader(root_path, training = True, batch_size = 1, shuffle=True, gt_downsample=True, pre_load=False)

data_loader_val = ImageDataLoader(root_path, training = False, batch_size = 1, shuffle=True, gt_downsample=True, pre_load=False)


def train_data():
    def reader():
        for blob in data_loader_train: # image num
            tmp_xs = blob["data"]
            tmp_1_ys = blob["gt_1_density"]
            tmp_2_ys = blob["gt_2_density"]
            # _vis_minibatch(tmp_xs)
            # _vis_minibatch(tmp_ys)
            yield tmp_xs, tmp_1_ys  # size    ((1,1,img.shape[0],img.shape[1])), (1,1,den.shape[0],den.shape[1])
    return reader()

def val_data():
    def reader():
        for blob in data_loader_val: # image num
            tmp_xs = blob["data"]
            tmp_1_ys = blob["gt_1_density"]
            tmp_2_ys = blob["gt_2_density"]
            # _vis_minibatch(tmp_xs)
            # _vis_minibatch(tmp_ys)
            yield tmp_xs, tmp_1_ys  # size    ((1,1,img.shape[0],img.shape[1])), (1,1,den.shape[0],den.shape[1])
    return reader()

#image_shape = [1, 128, 256]
#den_shape = [1, 32, 64]
image_shape = [3, 288, 512]
den_1_shape = [1, 72, 128]
#den_2_shape = [1, 288, 512]

image_val_shape = [3, 576, 1024]
den_val_1_shape = [1, 288, 512]
#den_val_2_shape = [1, 576, 1024]

# 定义输入数据大小，指定图像的形状，数据类型是浮点型
image = fluid.layers.data(name='image', shape=image_shape, dtype='float32')
# 定义标签，类型是整型
den_1 = fluid.layers.data(name='den_1', shape=den_1_shape, dtype='float32')
#den_2 = fluid.layers.data(name='den_2', shape=den_2_shape, dtype='float32')
# image_val = fluid.layers.data(name='image_val', shape=image_val_shape, dtype='float32')
# # 定义标签，类型是整型
# den_val = fluid.layers.data(name='den_val', shape=den_val_shape, dtype='float32')

#MCNN = mcnn_program(image)
#MCNN = inference_bn(image)
#MCNN = iterative_crowd_counting(image)
MCNN = Stacked_Pooling(image)
out_put_1 = MCNN.top_layer
#print(out_put_1.shape, out_put_2.shape)

etmap_num_1 = fluid.layers.reduce_sum(out_put_1)
den_gt_num_1 = fluid.layers.reduce_sum(den_1)


loss_1 = fluid.layers.square_error_cost(out_put_1, den_1)

avg_loss = fluid.layers.mean(loss_1)


optimizer = fluid.optimizer.Adam(learning_rate=1e-5)
#optimizer = fluid.optimizer.Momentum(learning_rate=1e-4, momentum=0.9)
#optimizer = fluid.optimizer.SGD(learning_rate=1e-4)
optimizer.minimize(avg_loss)

place = fluid.CUDAPlace(0)
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())

train_reader = paddle.batch(
        paddle.reader.shuffle(train_data, buf_size=3561),
        batch_size=BATCH_SIZE)

feeder = fluid.DataFeeder(place=place, feed_list=[image, den_1])

# feader_val = fluid.DataFeeder(place=place, feed_list=[image_val, den_val])

model_name = '_stack_pooling_dot_bbox'

output_dir = './saved_models/fluid' + model_name
mkdirs(output_dir)
train_check = output_dir + '/train/'
mkdirs(train_check)
test_check = output_dir + '/test/'
mkdirs(test_check)

"""
# #  load saved params
path = "/data1/ShengGe/ZSQ/paddlepaddle/saved_models/fluid_stack_pooling/train/epoch_000297/"  # finetune
print('loading...%s'%(path))
prog = fluid.default_main_program()
fluid.io.load_checkpoint(executor=exe, checkpoint_dir=path,
                          serial=0, main_program=prog)
"""

for pass_id in range(0,1000):
    for batch_id, data in enumerate(train_reader()):
        
        avg_loss_data_1, etmap_num_data_1, den_gt_num_data_1 = exe.run(
            fluid.default_main_program(),
            feed=feeder.feed(data),
            fetch_list=[avg_loss,etmap_num_1, den_gt_num_1])  # den_gt_num_1 = den_gt_num_2
            #fetch_list=[avg_loss, etmap_num, den_gt_num])  # den_gt_num_1 = den_gt_num_2
        if batch_id % 200 == 0:
            print('epoch: %05d  batch_id: %05d'%(pass_id, batch_id))
            print('avg_loss_1: %.7f etmap_num_1: %.7f den_gt_num_1:%.7f'%(avg_loss_data_1[0], etmap_num_data_1[0],
                                                                                    den_gt_num_data_1[0]))              
    
    
    # training
    print('saving.. epoch_%06d'%(pass_id))
    path = train_check + 'epoch_%06d'%(pass_id)
    mkdirs(path)
    prog = fluid.default_main_program()
    trainer_args = {"epoch_id": pass_id,"step_id": 100}
    fluid.io.save_checkpoint(executor=exe,
                            checkpoint_dir=path,
                            trainer_id=0,
                            trainer_args=trainer_args,
                            main_program=prog,
                            max_num_checkpoints=3)
    #test
    param_path = test_check + 'epoch_%06d'%(pass_id)
    mkdirs(param_path)
    prog = fluid.default_main_program()
    fluid.io.save_params(executor=exe, dirname=param_path, main_program=prog)


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



