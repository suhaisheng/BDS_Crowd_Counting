# -*- coding:utf-8 -*-
from __future__ import print_function
import paddle.v2 as paddle
import paddle.fluid as fluid
import paddle


def conv( input, num_filters, filter_size, stride, act=None):
	if act is not None:
		conv_layer = fluid.layers.conv2d(input=input, num_filters=num_filters,filter_size=filter_size,stride=stride, 
			padding=(int(( filter_size- 1) / 2),int((filter_size - 1) / 2)),
			act=act, param_attr=fluid.initializer.Normal(loc=0.0, scale=0.01),
			bias_attr = fluid.initializer.Constant(value=0.0)
			)
	else:
		conv_layer = fluid.layers.conv2d(name=name,input=input, num_filters=num_filters,filter_size=filter_size,stride=stride, 
			padding=(int(( filter_size- 1) / 2),int((filter_size - 1) / 2)),
		    param_attr=fluid.initializer.Normal(loc=0.0, scale=0.01),
			bias_attr = fluid.initializer.Constant(value=0.0)
			)
	return conv_layer 



class mcnn_program():
	def __init__(self,img):
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


