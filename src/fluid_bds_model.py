# -*- coding:utf-8 -*-
from __future__ import print_function
import paddle.v2 as paddle
import paddle.fluid as fluid
import paddle


def conv( input, num_filters, filter_size, stride, act):
	conv_layer = fluid.layers.conv2d( input=input, num_filters=num_filters,filter_size=filter_size,stride=stride, 
		padding=(int(( filter_size- 1) / 2),int((filter_size - 1) / 2)),
		act=act, param_attr=fluid.initializer.Normal(loc=0.0, scale=0.01),
		bias_attr = fluid.initializer.Constant(value=0.0))

	return conv_layer 

def multi_scale_block(pre, in_con, out_dim, is_bn=False):
    conv_9 = conv(name=pre+'_conv9', input=in_con, num_filters=out_dim, filter_size=9, stride=1, act="relu")
    conv_7 = conv(name=pre+'_conv7', input=in_con, num_filters=out_dim, filter_size=7, stride=1, act="relu")
    conv_5 = conv(name=pre+'_conv5', input=in_con, num_filters=out_dim, filter_size=5, stride=1, act="relu")
    conv_3 = conv(name=pre+'_conv3', input=in_con, num_filters=out_dim, filter_size=3, stride=1, act="relu")
    concat = fluid.layers.concat(input=[conv_9, conv_7, conv_5, conv_3], axis=1)  # [NCHW]
    #print(concat.shape)
    #biases = fluid.layers.create_parameter(shape=[out_dim*4], dtype='float32', name=pre+'_biases', is_bias=True)
    #bias = fluid.layers.elementwise_add(x=concat,y=biases)
    #bias = concat + biases
    #biases = fluid.ParamAttr(name=pre+'biases', regularizer=fluid.regularizer.L2DecayRegularizer(
    #    regularization_coeff=0.1),trainable=True)
    #bias = fluid.layers.fc(input=concat, size=[out_dim*4], bias_attr=biases)
    bias = conv(name=pre+'_fuse', input=concat, num_filters=1,filter_size=1,stride=1,act=None)

    if is_bn:
        bias = fluid.layers.batch_norm(input=bias)

    msb = fluid.layers.relu(bias)
    #print("msb",msb.shape)
    return msb

## MSCNN model
class inference_bn():
    def __init__(self, img):
    	#print(img.shape)
    	self.con1 = conv(name='con1',input=img,num_filters=64, filter_size=9,stride=1,act="relu")  # in_dim=3
    	self.msb_con2 = multi_scale_block('msb_con2', self.con1, 16, is_bn=True)
    	self.pool_msb_con2 = fluid.layers.pool2d(input=self.msb_con2, pool_size=2, pool_type='max',pool_stride=2,global_pooling=False)
    	self.msb_con3 = multi_scale_block('msb_con3', self.pool_msb_con2, 32, is_bn=True)
    	self.msb_con4 = multi_scale_block('msb_con4', self.msb_con3, 32, is_bn=True)
    	self.pool_msb_con4 = fluid.layers.pool2d(input=self.msb_con4, pool_size=2, pool_type='max',pool_stride=2,global_pooling=False)
    	self.msb_con5 = multi_scale_block('msb_con5', self.pool_msb_con4, 64, is_bn=True)
    	self.msb_con6 = multi_scale_block('msb_con6', self.msb_con5, 64, is_bn=True)
    	self.mpl_con7 = conv(name='mpl_con7',input=self.msb_con6,num_filters=1000, filter_size=1, stride=1, act="relu")
    	self.con_out = conv(name='con_out',input=self.mpl_con7,num_filters=1, filter_size=1, stride=1, act="sigmoid")
    	self.image_out = self.con_out
    	#print(self.image_out.shape)

    @property
    def top_layer(self):
    	print(self.image_out.shape)
    	return self.image_out #[1,1,h,w]

## MCNN model + Attention model
class mcnn_program():
	def __init__(self, img):
		wd = 256
		ht = 128
		self.conv1_1 = conv(input=img, num_filters=16,filter_size=9,stride=1,act="relu")
		self.pool1_1 = fluid.layers.pool2d(input=self.conv1_1, pool_size=2, pool_type='max',pool_stride=2,global_pooling=False)
		self.conv1_2 = conv(input=self.pool1_1, num_filters=32,filter_size=7,stride=1,act="relu")
		self.pool1_2 = fluid.layers.pool2d(input=self.conv1_2, pool_size=2, pool_type='max',pool_stride=2,global_pooling=False)
		self.conv1_3 = conv(input=self.pool1_2, num_filters=16,filter_size=7,stride=1,act="relu")
		self.conv1_4 = conv(input=self.conv1_3, num_filters=8,filter_size=7,stride=1,act="relu")
		"""
		self.attention_1 = conv(input=self.conv1_3, num_filters=8, filter_size=1, stride=1, act="relu")
		self.attention_1 = fluid.layers.reshape(self.attention_1, [-1, ht * wd])
		self.attention_1 = fluid.layers.softmax(self.attention_1)
		#self.attention_1 = fluid.layers.sigmoid(self.attention_1)
		self.attention_1 = fluid.layers.reshape(self.attention_1,[-1, 8, ht, wd])
		self.branch_1 = self.attention_1 * self.conv1_4
		"""
		self.conv2_1 = conv(input=img, num_filters=20,filter_size=7,stride=1,act="relu")
		self.pool2_1 = fluid.layers.pool2d(input=self.conv2_1, pool_size=2, pool_type='max',pool_stride=2,global_pooling=False)
		self.conv2_2 = conv(input=self.pool2_1, num_filters=40,filter_size=5,stride=1,act="relu")
		self.pool2_2 = fluid.layers.pool2d(input=self.conv2_2, pool_size=2, pool_type='max',pool_stride=2,global_pooling=False)
		self.conv2_3 = conv(input=self.pool2_2, num_filters=20,filter_size=5,stride=1,act="relu")
		self.conv2_4 = conv(input=self.conv2_3, num_filters=10,filter_size=5,stride=1,act="relu")
		"""
		self.attention_2 = conv(input=self.conv2_3, num_filters=10, filter_size=1, stride=1, act="relu")
		self.attention_2 = fluid.layers.reshape(self.attention_2, [-1, ht * wd])
		self.attention_2 = fluid.layers.softmax(self.attention_2)
		#self.attention_2 = fluid.layers.sigmoid(self.attention_2)
		self.attention_2 = fluid.layers.reshape(self.attention_2,[-1, 10, ht, wd])
		self.branch_2 = self.attention_2 * self.conv2_4
		"""
		self.conv3_1 = conv(input=img, num_filters=24,filter_size=5,stride=1,act="relu")
		self.pool3_1 = fluid.layers.pool2d(input=self.conv3_1, pool_size=2, pool_type='max',pool_stride=2,global_pooling=False)
		self.conv3_2 = conv(input=self.pool3_1, num_filters=48,filter_size=3,stride=1,act="relu")
		self.pool3_2 = fluid.layers.pool2d(input=self.conv3_2, pool_size=2, pool_type='max',pool_stride=2,global_pooling=False)
		self.conv3_3 = conv(input=self.pool3_2, num_filters=24,filter_size=3,stride=1,act="relu")
		self.conv3_4 = conv(input=self.conv3_3, num_filters=12,filter_size=3,stride=1,act="relu")
		"""
		self.attention_3 = conv(input=self.conv3_3, num_filters=12, filter_size=1, stride=1, act="relu")
		self.attention_3 = fluid.layers.reshape(self.attention_3, [-1, ht * wd])
		self.attention_3 = fluid.layers.softmax(self.attention_3)
		#self.attention_3 = fluid.layers.sigmoid(self.attention_3)
		self.attention_3 = fluid.layers.reshape(self.attention_3,[-1, 12, ht, wd])
		self.branch_3 = self.attention_3 * self.conv3_4
		"""
		self.top_layers = []
		#self.top_layers.append(self.branch_1)
		#self.top_layers.append(self.branch_2)
		#self.top_layers.append(self.branch_3)
		self.top_layers.append(self.conv1_4)
		self.top_layers.append(self.conv2_4)
		self.top_layers.append(self.conv3_4)
		self.concat = fluid.layers.concat(input=self.top_layers, axis=1)
		
		self.fuse = conv(input=self.concat, num_filters=1,filter_size=1,stride=1,act="relu")
		print(self.fuse.shape)
		#self.attention = fluid.layers.reshape(self.attention, [-1,128*256])
		
		#self.attention = fluid.layers.sigmoid(self.attention)
		#self.attention = fluid.layers.reshape(self.attention,[-1, 1, 128, 256])
		#print(self.attention.shape)
		#print(self.fuse.shape)

		#self.final = self.attention * self.fuse
		#self.output = conv(name='output',input=self.final, num_filters=1, filter_size=1, stride=1) # act? output
		#self.output=fluid.layers.conv2d(name='output',input=self.final, num_filters=1,filter_size=1,stride=1, 
		#	padding=(0,0),
			#act="relu",
		#	param_attr=fluid.initializer.Normal(loc=0.0, scale=0.01),
		#	bias_attr = fluid.initializer.Constant(value=0.0))
		
	@property
	def top_layer(self):
		return self.fuse
		#return self.output


## Iterative crowd counting model
class iterative_crowd_counting():
	def __init__(self, img):
		wd = 512
		ht = 288
		## first stage
		## LR-CNN
		self.conv1_1 = conv(input=img,num_filters=64,filter_size=3,stride=1,act="relu")
		self.conv1_2 = conv(input=self.conv1_1,num_filters=64,filter_size=3,stride=1,act="relu")
		self.pool1_1 = fluid.layers.pool2d(input=self.conv1_2, pool_size=2, pool_type='max',pool_stride=2,global_pooling=False)
		self.conv1_3 = conv(input=self.pool1_1,num_filters=128,filter_size=3,stride=1,act="relu")
		self.conv1_4 = conv(input=self.conv1_3,num_filters=128,filter_size=3,stride=1,act="relu")
		self.pool1_2 = fluid.layers.pool2d(input=self.conv1_4, pool_size=2, pool_type='max',pool_stride=2,global_pooling=False)
		self.conv1_5 = conv(input=self.pool1_2,num_filters=256,filter_size=3,stride=1,act="relu")

		self.conv1_6 = conv(input=self.conv1_5,num_filters=256,filter_size=3,stride=1,act="relu")
		self.conv1_7 = conv(input=self.conv1_6,num_filters=256,filter_size=3,stride=1,act="relu")
		self.conv1_8 = conv(input=self.conv1_7,num_filters=196,filter_size=7,stride=1,act="relu")
		self.conv1_9 = conv(input=self.conv1_8,num_filters=96,filter_size=5,stride=1,act="relu")
		self.conv1_10= conv(input=self.conv1_9,num_filters=32,filter_size=3,stride=1,act="relu")
		self.conv1_11 = conv(input=self.conv1_10,num_filters=1,filter_size=1,stride=1,act="relu")   #low resolution prediction maps

		## HR-CNN
		self.conv2_1 = conv(input=img,num_filters=16,filter_size=7,stride=1,act="relu")
		self.pool2_1 = fluid.layers.pool2d(input=self.conv2_1,pool_size=2,pool_type='max',pool_stride=2,global_pooling=False)
		self.conv2_2 = conv(input=self.pool2_1,num_filters=24,filter_size=5,stride=1,act="relu")
		self.pool2_2 = fluid.layers.pool2d(input=self.conv2_2,pool_size=2,pool_type='max',pool_stride=2,global_pooling=False)
		self.conv2_3 = conv(input=self.pool2_2,num_filters=48,filter_size=3,stride=1,act="relu")
		self.conv2_4 = conv(input=self.conv2_3,num_filters=48,filter_size=3,stride=1,act="relu")
		self.conv2_5 = conv(input=self.conv2_4,num_filters=24,filter_size=3,stride=1,act="relu")

		self.top_layers = []
		self.top_layers.append(self.conv2_5) # [1,1,h,w]
		self.top_layers.append(self.conv1_5) # conv1_7
		self.top_layers.append(self.conv1_11)
		self.concat = fluid.layers.concat(input=self.top_layers, axis=1)


		self.conv2_6 = conv(input=self.concat,num_filters=196,filter_size=7,stride=1,act="relu")
		self.conv2_7 = conv(input=self.conv2_6,num_filters=96,filter_size=5,stride=1,act="relu")
		self.upsample_1 = fluid.layers.resize_bilinear(input=self.conv2_7,out_shape=[ht/2,wd/2])
		#self.upsample_1 = fluid.layers.image_resize(input=self.conv2_7,out_shape=[ht,wd])
		self.conv2_8 = conv(input=self.upsample_1,num_filters=32,filter_size=3,stride=1,act="relu")
		#self.upsample_2 = fluid.layers.image_resize(input=self.conv2_8,out_shape=[ht,wd])
		self.upsample_2 = fluid.layers.resize_bilinear(input=self.conv2_8,out_shape=[ht,wd])
		self.conv2_9 = conv(input=self.upsample_2,num_filters=1,filter_size=1,stride=1,act="relu")   # high resolution prediction maps
		
		## second stage
		## LR-CNN
		self.top_layers = []
		self.top_layers.append(img)
		self.top_layers.append(self.conv2_9)
		self.concat = fluid.layers.concat(input=self.top_layers, axis=1)
		self.conv3_1 = conv(input=self.concat,num_filters=64,filter_size=3,stride=1,act="relu")
		self.conv3_2 = conv(input=self.conv3_1,num_filters=64,filter_size=3,stride=1,act="relu")
		self.pool3_1 = fluid.layers.pool2d(input=self.conv3_2, pool_size=2, pool_type='max',pool_stride=2,global_pooling=False)
		self.conv3_3 = conv(input=self.pool3_1,num_filters=128,filter_size=3,stride=1,act="relu")
		self.conv3_4 = conv(input=self.conv3_3,num_filters=128,filter_size=3,stride=1,act="relu")
		self.pool3_2 = fluid.layers.pool2d(input=self.conv3_4, pool_size=2, pool_type='max',pool_stride=2,global_pooling=False)
		self.conv3_5 = conv(input=self.pool3_2,num_filters=256,filter_size=3,stride=1,act="relu")
		self.top_layers = []
		self.top_layers.append(self.conv3_5)
		self.top_layers.append(self.conv1_11)
		self.concat = fluid.layers.concat(input=self.top_layers, axis=1)
		self.conv3_6 = conv(input=self.concat,num_filters=256,filter_size=3,stride=1,act="relu")
		self.conv3_7 = conv(input=self.conv3_6,num_filters=256,filter_size=3,stride=1,act="relu")
		self.conv3_8 = conv(input=self.conv3_7,num_filters=196,filter_size=7,stride=1,act="relu")
		self.conv3_9 = conv(input=self.conv3_8,num_filters=96,filter_size=5,stride=1,act="relu")
		self.conv3_10= conv(input=self.conv3_9,num_filters=32,filter_size=3,stride=1,act="relu")
		self.conv3_11 = conv(input=self.conv3_10,num_filters=1,filter_size=1,stride=1,act="relu")   #low resolution prediction maps

		## HR-CNN
		self.top_layers = []
		self.top_layers.append(img)
		self.top_layers.append(self.conv2_9)
		self.concat = fluid.layers.concat(input=self.top_layers, axis=1)
		self.conv4_1 = conv(input=self.concat,num_filters=16,filter_size=7,stride=1,act="relu")
		self.pool4_1 = fluid.layers.pool2d(input=self.conv4_1,pool_size=2,pool_type='max',pool_stride=2,global_pooling=False)
		self.conv4_2 = conv(input=self.pool4_1,num_filters=24,filter_size=5,stride=1,act="relu")
		self.pool4_2 = fluid.layers.pool2d(input=self.conv4_2,pool_size=2,pool_type='max',pool_stride=2,global_pooling=False)
		self.conv4_3 = conv(input=self.pool4_2,num_filters=48,filter_size=3,stride=1,act="relu")
		self.conv4_4 = conv(input=self.conv4_3,num_filters=48,filter_size=3,stride=1,act="relu")
		self.conv4_5 = conv(input=self.conv4_4,num_filters=24,filter_size=3,stride=1,act="relu")

		self.top_layers = []
		self.top_layers.append(self.conv4_5) # [1,1,h,w]
		self.top_layers.append(self.conv3_6) # conv3_7
		self.top_layers.append(self.conv3_11)
		self.concat = fluid.layers.concat(input=self.top_layers, axis=1)


		self.conv4_6 = conv(input=self.concat,num_filters=196,filter_size=7,stride=1,act="relu")
		self.conv4_7 = conv(input=self.conv4_6,num_filters=96,filter_size=5,stride=1,act="relu")
		self.upsample_3 = fluid.layers.resize_bilinear(input=self.conv4_7,out_shape=[ht/2,wd/2])
		#self.upsample_3 = fluid.layers.image_resize(input=self.conv4_7,out_shape=[ht,wd])
		self.conv4_8 = conv(input=self.upsample_3,num_filters=32,filter_size=3,stride=1,act="relu")
		#self.upsample_4 = fluid.layers.image_resize(input=self.conv4_8,out_shape=[ht,wd])
		self.upsample_4 = fluid.layers.resize_bilinear(input=self.conv4_8,out_shape=[ht,wd])
		self.conv4_9 = conv(input=self.upsample_4,num_filters=1,filter_size=1,stride=1,act="relu")   # high resolution prediction maps
		
	@property
	def top_layer(self):
		#return self.conv1_11, self.conv2_9
		return self.conv3_11, self.conv4_9

## Stacked pooling model
class Stacked_Pooling():
	def __init__(self, img):
		## Multi-kernel pooling  && Stacked pooling
		paddings = [0, 0, 0, 0, 0, 1, 0, 1]
		self.conv1 = conv(input=img,num_filters=64,filter_size=5,stride=1,act="relu")
		self.conv2 = conv(input=self.conv1,num_filters=64,filter_size=5,stride=1,act="relu")
		self.pool1 = fluid.layers.pool2d(input=self.conv2,pool_size=2,pool_type='max',pool_stride=2,global_pooling=False)
		self.pool2 = fluid.layers.pool2d(input=self.pool1,pool_size=2,pool_type='max',pool_stride=1,global_pooling=False)
		self.pool2 = fluid.layers.pad(self.pool2, paddings)
		self.pool3 = fluid.layers.pool2d(input=self.pool2,pool_size=3,pool_type='max',pool_stride=1,pool_padding=1,global_pooling=False)
		print (img.shape, self.pool1.shape, self.pool2.shape, self.pool3.shape)
		
		self.concat = (self.pool1 +self.pool2 + self.pool3)/3.0
		print(self.concat.shape)
		##(-1, 144, 256)

		self.conv3 = conv(input=self.concat,num_filters=128,filter_size=5,stride=1,act="relu")
		self.conv4 = conv(input=self.conv3,num_filters=128,filter_size=5,stride=1,act="relu")
		self.pool4 = fluid.layers.pool2d(input=self.conv4,pool_size=2,pool_type='max',pool_stride=2,global_pooling=False)
		self.pool5 = fluid.layers.pool2d(input=self.pool4,pool_size=2,pool_type='max',pool_stride=1,global_pooling=False)
		self.pool5 = fluid.layers.pad(self.pool5, paddings)
		self.pool6 = fluid.layers.pool2d(input=self.pool5,pool_size=3,pool_type='max',pool_stride=1,pool_padding=1,global_pooling=False)

		self.concat = (self.pool4 +self.pool5 + self.pool6)/3.0
		print(self.concat.shape)
		
		self.conv5 = conv(input=self.concat,num_filters=256,filter_size=3,stride=1,act="relu")
		self.conv6 = conv(input=self.conv5,num_filters=256,filter_size=3,stride=1,act="relu")
		self.pool7 = fluid.layers.pool2d(input=self.conv4,pool_size=2,pool_type='max',pool_stride=2,global_pooling=False)
		self.pool8 = fluid.layers.pool2d(input=self.pool7,pool_size=2,pool_type='max',pool_stride=1,global_pooling=False)
		self.pool8 = fluid.layers.pad(self.pool8, paddings)
		self.pool9 = fluid.layers.pool2d(input=self.pool8,pool_size=3,pool_type='max',pool_stride=1,pool_padding=1,global_pooling=False)

		self.concat = (self.pool7 +self.pool8 + self.pool9)/3.0
		print(self.concat.shape)

		self.conv7 = conv(input=self.concat,num_filters=128,filter_size=3,stride=1,act="relu")
		self.conv8 = conv(input=self.conv7,num_filters=64,filter_size=3,stride=1,act="relu")
		self.conv9 = conv(input=self.conv8,num_filters=32,filter_size=3,stride=1,act="relu")
		self.conv10 = conv(input=self.conv9,num_filters=16,filter_size=3,stride=1,act="relu")
		self.conv11 = conv(input=self.conv10,num_filters=1,filter_size=1,stride=1,act="relu")

	@property
	def top_layer(self):
		return self.conv11







