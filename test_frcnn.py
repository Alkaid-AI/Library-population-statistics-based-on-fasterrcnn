from __future__ import division
import os
import cv2
import numpy as np
import sys
import pickle
from optparse import OptionParser
import time
from keras_frcnn import config
from keras import backend as K
from keras.layers import Input
from keras.models import Model
from keras_frcnn import roi_helpers

sys.setrecursionlimit(40000)

parser = OptionParser()


# =========笔记：


# =========================设置测试图片的路径=============================================

parser.add_option("-p", "--path", dest="test_path", help="Path to test data.",              #dest代表存储的变量
				  default="D:\\AI_path\\keras-frcnn-master\\test_images")
parser.add_option("-n", "--num_rois", type="int", dest="num_rois",
				  help="Number of ROIs per iteration. Higher means more memory use.", default=32)
parser.add_option("--config_filename", dest="config_filename", help=
"Location to read the metadata related to the training (generated when training).",
				  default="config.pickle")          # 这个是之前训练的成果(训练时产生的元数据)的存放位置。
parser.add_option("--network", dest="network", help="Base network to use. Supports vgg or resnet50.",
				  default='resnet50')

# ============================ parse_args() 返回的两个值：=======================================
				# options，它是一个对象（optpars.Values），保存有命令行参数值。只要知道命令行参数名，就可以访问其对应的值 。
				# args，它是一个由 positional arguments 组成的列表。
(options, args) = parser.parse_args()       # 通过parse_args()函数的解析，获得选项，如options.test_path


if not options.test_path:  # if filename is not given
	parser.error('Error: path to test data must be specified. Pass --path to command line') #用户也可以使用 parser.error() 方法来自定义部分异常的处理：

config_output_filename = options.config_filename

with open(config_output_filename, 'rb') as f_in:
	C = pickle.load(f_in)                               # pickle用于python特有的类型，和python的数据类型间进行转换，提供四个功能 dumps,dump,loads,load.
														#pickle模块是以二进制的形式序列化后保存到文件中
														# pickle.dumps 将数据通过特殊的形式转换为只有python语言认识的字符串，pickle.dumps()方法跟pickle.dump()方法
														# 的区别在于，pickle.dumps()方法不需要写入文件中，它是直接返回一个序列化的bytes对象。
														# 将序列化的对象从文件file中读取出来，pickle.loads()方法跟pickle.load()方法的区别在于，
														# pickle.loads()方法是直接从bytes对象中读取序列化的信息，而非从文件中读取。
if C.network == 'resnet50':                             # 如果之前使用的network是 resnet50的话....
	import keras_frcnn.resnet as nn
elif C.network == 'vgg':
	import keras_frcnn.vgg as nn

# turn off any data augmentation at test time
C.use_horizontal_flips = False
C.use_vertical_flips = False
C.rot_90 = False

img_path = options.test_path


def format_img_size(img, C):                         # 参数C代表之前的训练的参数
	""" formats the image size based on config """
	img_min_side = float(C.im_size)                  #
	(height, width, _) = img.shape

	if width <= height:
		ratio = img_min_side / width
		new_height = int(ratio * height)
		new_width = int(img_min_side)
	else:
		ratio = img_min_side / height
		new_width = int(ratio * width)
		new_height = int(img_min_side)
	img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
	return img, ratio                          			# 得到训练出的参数所需要的图片的尺寸 与 我们输入需要检测的图片尺寸之比ratio，以及得到更改过的与参数适配的图片


def format_img_channels(img, C):                         # 将我们的图片的三个通道
	""" formats the image channels based on config """
	img = img[:, :, (2, 1, 0)]
	img = img.astype(np.float32)                        # astype转换数据类型为32位浮点型
	img[:, :, 0] -= C.img_channel_mean[0]				#将我们的图片的三个通道格式化为与C的三个通道的均值。
	img[:, :, 1] -= C.img_channel_mean[1]
	img[:, :, 2] -= C.img_channel_mean[2]
	img /= C.img_scaling_factor
	img = np.transpose(img, (2, 0, 1))                 # numpy的transpose是转置（二维情况是转置很好理解，三维的时候比如这里就是把0轴和1轴相互换一下）
	img = np.expand_dims(img, axis=0)
	return img


def format_img(img, C):
	""" formats an image for model prediction based on config """
	img, ratio = format_img_size(img, C)
	img = format_img_channels(img, C)
	return img, ratio


# Method to transform the coordinates of the bounding box to its original size    将界定框更改到它原来的尺寸。
def get_real_coordinates(ratio, x1, y1, x2, y2):
	real_x1 = int(round(x1 // ratio))
	real_y1 = int(round(y1 // ratio))
	real_x2 = int(round(x2 // ratio))
	real_y2 = int(round(y2 // ratio))

	return (real_x1, real_y1, real_x2, real_y2)                               # 得到实际的两个坐标点


class_mapping = C.class_mapping

if 'bg' not in class_mapping:
	class_mapping['bg'] = len(class_mapping)

class_mapping = {v: k for k, v in class_mapping.items()}
print(class_mapping)
class_to_color = {class_mapping[v]: np.random.randint(0, 255, 3) for v in class_mapping}
C.num_rois = int(options.num_rois)

# 是不是得有这个声明，下面的那个num_features的错误才会消失？？？？
num_features = []

if C.network == 'resnet50':
	num_features = 1024
elif C.network == 'vgg':
	num_features = 512



if K.image_dim_ordering() == 'th':
	input_shape_img = (3, None, None)
	input_shape_features = (num_features, None, None)
else:
	input_shape_img = (None, None, 3)
	input_shape_features = (None, None, num_features)

img_input = Input(shape=input_shape_img)
roi_input = Input(shape=(C.num_rois, 4))
feature_map_input = Input(shape=input_shape_features)

# define the base network (resnet here, can be VGG, Inception, etc)
shared_layers = nn.nn_base(img_input, trainable=True)

# define the RPN, built on the base layers
num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
rpn_layers = nn.rpn(shared_layers, num_anchors)

classifier = nn.classifier(feature_map_input, roi_input, C.num_rois, nb_classes=len(class_mapping), trainable=True)

model_rpn = Model(img_input, rpn_layers)
model_classifier_only = Model([feature_map_input, roi_input], classifier)

model_classifier = Model([feature_map_input, roi_input], classifier)

print('Loading weights from {}'.format(C.model_path))
model_rpn.load_weights(C.model_path, by_name=True)
model_classifier.load_weights(C.model_path, by_name=True)

model_rpn.compile(optimizer='sgd', loss='mse')
model_classifier.compile(optimizer='sgd', loss='mse')

all_imgs = []

classes = {}

bbox_threshold = 0.55          # 这个值越低 出现的不确定框越多，高一点会比较准确

visualise = True





for idx, img_name in enumerate(sorted(os.listdir(img_path))):
	if not img_name.lower().endswith(('.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff')):
		continue
	print(img_name)
	st = time.time()
	filepath = os.path.join(img_path, img_name)

	img = cv2.imread(filepath)
#                                                                     调整原始的图像（可变）
	img = cv2.resize(img, (720, 480))

	X, ratio = format_img(img, C)

	if K.image_dim_ordering() == 'tf':
		X = np.transpose(X, (0, 2, 3, 1))

	# get the feature maps and output from the RPN
	[Y1, Y2, F] = model_rpn.predict(X)

	R = roi_helpers.rpn_to_roi(Y1, Y2, C, K.image_dim_ordering(), overlap_thresh=0.7)

	# convert from (x1,y1,x2,y2) to (x,y,w,h)
	R[:, 2] -= R[:, 0]
	R[:, 3] -= R[:, 1]

	# apply the spatial pyramid pooling to the proposed regions
	bboxes = {}
	probs = {}

	for jk in range(R.shape[0] // C.num_rois + 1):
		ROIs = np.expand_dims(R[C.num_rois * jk:C.num_rois * (jk + 1), :], axis=0)
		if ROIs.shape[1] == 0:
			break

		if jk == R.shape[0] // C.num_rois:
			# pad R
			curr_shape = ROIs.shape
			target_shape = (curr_shape[0], C.num_rois, curr_shape[2])
			ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
			ROIs_padded[:, :curr_shape[1], :] = ROIs
			ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
			ROIs = ROIs_padded

		[P_cls, P_regr] = model_classifier_only.predict([F, ROIs])

		for ii in range(P_cls.shape[1]):

			if np.max(P_cls[0, ii, :]) < bbox_threshold or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
				continue

			cls_name = class_mapping[np.argmax(P_cls[0, ii, :])]

			if cls_name not in bboxes:
				bboxes[cls_name] = []
				probs[cls_name] = []

			(x, y, w, h) = ROIs[0, ii, :]

			cls_num = np.argmax(P_cls[0, ii, :])
			try:
				(tx, ty, tw, th) = P_regr[0, ii, 4 * cls_num:4 * (cls_num + 1)]
				tx /= C.classifier_regr_std[0]
				ty /= C.classifier_regr_std[1]
				tw /= C.classifier_regr_std[2]
				th /= C.classifier_regr_std[3]
				x, y, w, h = roi_helpers.apply_regr(x, y, w, h, tx, ty, tw, th)
			except:
				pass
			bboxes[cls_name].append(
				[C.rpn_stride * x, C.rpn_stride * y, C.rpn_stride * (x + w), C.rpn_stride * (y + h)])
			probs[cls_name].append(np.max(P_cls[0, ii, :]))

	all_dets = []

	for key in bboxes:                                     # key 是类别数，如我们这里只检测人，key 就只有一类：person
		bbox = np.array(bboxes[key])

		new_boxes, new_probs = roi_helpers.non_max_suppression_fast(bbox, np.array(probs[key]), overlap_thresh=0.5)

# '''在图片的右下角输出人数统计'''
		height = img.shape[0]
		width = img.shape[1]
		textLabel1 = 'Total:{}'.format(new_boxes.shape[0])
		textOrg1 = (width - 200, height - 30)
		cv2.putText(img, textLabel1, textOrg1, cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 50), 2)

		for jk in range(new_boxes.shape[0]):                              # new_boxes.shape[0] 就是检测出来的同一类的目标的数目！！！！
			(x1, y1, x2, y2) = new_boxes[jk, :]

			(real_x1, real_y1, real_x2, real_y2) = get_real_coordinates(ratio, x1, y1, x2, y2)

# 对应的包含人像的矩形框，重要！！！
			cv2.rectangle(img, (real_x1, real_y1), (real_x2, real_y2),
						  (int(class_to_color[key][0]), int(class_to_color[key][1]), int(class_to_color[key][2])), 2)
			# aaa = aaa+1
			textLabel = '{}: {}'.format(key, int(100 * new_probs[jk]))              # 对标签进行更改，person：98
			all_dets.append((key, 100 * new_probs[jk]))

# 在绘制文字之前，使用getTextSize()接口先获取待绘制文本框的大小，以方便放置文本框，功能：计算文本的宽和高，输出retval是文本框的宽和高，retval[0],retval[1]代表宽和高
# baseLine，基线相对于最底层文本的y坐标；
			(retval, baseLine) = cv2.getTextSize(textLabel, cv2.FONT_HERSHEY_COMPLEX, 1, 1)
			textOrg = (real_x1, real_y1 - 0)
# rectangle(ima,左上角的点，右下角的点，颜色(B,G,R), 矩形边框的厚度(若为负值则表示填充整个矩形))
# 			cv2.rectangle(img, (textOrg[0] - 5, textOrg[1] + baseLine - 5),
# 						  (textOrg[0] + retval[0] + 5 , textOrg[1] - retval[1] - 5), (255, 255, 255), 2)          # 对应？这一步的意义在哪？ 可以不要的！！
			cv2.rectangle(img, (textOrg[0] - 5, textOrg[1] + baseLine - 5),
						  (textOrg[0] + retval[0] + 5, textOrg[1] - retval[1] - 5), (255, 255, 255), -1)    # 对应白色文字框
			# 分别对应 img:待绘制的图像，textLabel:待绘制的文字，textOrg：文本框左下角的坐标，字体类型，字体缩放倍数，字体颜色，字体粗细
			cv2.putText(img, textLabel, textOrg, cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 1)



	print('Elapsed time = {}'.format(time.time() - st))
	print(all_dets)
	cv2.imshow('img', img)
	cv2.waitKey(0)
	#  ======================在进行测试时候要把最后一行的注释去掉,用来存储已经预测完的图片===========================
	cv2.imwrite('./results_imgs/{}.jpg'.format(idx), img)


