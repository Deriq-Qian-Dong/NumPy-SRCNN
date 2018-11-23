# -*- coding: utf-8 -*-
"""
Created on 2018/11/20 21:51
@author: Sucre
@email: qian.dong.2018@gmail.com
"""
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
import math
import time
import cv2
import os


class MSE(object):
    def __init__(self):
        print("MSE")

    def cal_loss(self, x, y):
        '''
        :param x: train输入模型计算得到的out
        :param y: target
        :return: 均方差loss
        '''
        shape = x.shape
        N = 1
        for i in shape:
            N = N * i
        loss = np.sum(np.square(y - x)) / N
        return loss

    def gradient(self, x, y):
        '''
        :param x: train输入模型计算得到的out
        :param y: target
        :return: mse导数
        '''
        dx = -(y - x)
        return dx


class Relu(object):
    def __init__(sel):
        print("Relu")

    def forward(self, x):
        '''
        :param x: 待激活的向量
        :return: 激活后的向量
        '''
        self.x = x
        return np.maximum(x, 0)

    def backward(self):
        pass


class Conv2D(object):
    def __init__(self, shape, input_channels, output_channels, ksize=3, stride=1, method='SAME'):
        '''
        :param shape: 最先输入矩阵的shape
        :param input_channels: 输入矩阵的channel
        :param output_channels: 输出矩阵的channel
        :param ksize: kernel 的长度
        :param stride: 步长
        :param method: 卷积的形式
        '''
        self.output_channels = output_channels
        self.input_channels = input_channels
        self.batchsize = 1
        self.stride = stride
        self.ksize = ksize
        self.method = method
        # He normal参数初始化
        fan_in = reduce(lambda x, y: x * y, shape)
        stddev = math.sqrt(2 / fan_in)
        self.weights = np.random.normal(0, stddev, (ksize, ksize, self.input_channels, self.output_channels))
        self.bias = np.random.normal(0, stddev, self.output_channels)
        self.w_gradient = np.zeros(self.weights.shape)
        self.b_gradient = np.zeros(self.bias.shape)

    def forward(self, x):
        '''
        :param x: 输入卷积层的矩阵
        :return: 输出卷积层的矩阵
        '''
        shape = x.shape
        self.input_shape = shape
        col_weights = self.weights.reshape([-1, self.output_channels])
        if self.method == 'VALID':
            self.eta = np.zeros(
                (shape[0], (shape[1] - self.ksize + 1) // self.stride, (shape[1] - self.ksize + 1) // self.stride,
                 self.output_channels))
        if self.method == 'SAME':
            self.eta = np.zeros((shape[0], shape[1] // self.stride, shape[2] // self.stride, self.output_channels))
        if self.method == 'SAME':
            x = np.pad(x, (
                (0, 0), (self.ksize // 2, self.ksize // 2), (self.ksize // 2, self.ksize // 2), (0, 0)),
                       'constant', constant_values=0)
        self.output_shape = self.eta.shape

        self.col_image = []
        conv_out = np.zeros(self.eta.shape)
        for i in range(self.batchsize):
            img_i = x[i][np.newaxis, :]
            self.col_image_i = im2col(img_i, self.ksize, self.stride)
            conv_out[i] = np.reshape(np.dot(self.col_image_i, col_weights) + self.bias, self.eta[0].shape)
            self.col_image.append(self.col_image_i)
        self.col_image = np.array(self.col_image)
        return conv_out

    def gradient(self, eta):
        '''
        计算网络中的梯度
        '''
        self.eta = eta
        col_eta = np.reshape(eta, [self.batchsize, -1, self.output_channels])

        for i in range(self.batchsize):
            self.w_gradient += np.dot(self.col_image[i].T, col_eta[i]).reshape(self.weights.shape)
        self.b_gradient += np.sum(col_eta, axis=(0, 1))

        # deconv of padded eta with flippd kernel to get next_eta
        if self.method == 'VALID':
            pad_eta = np.pad(self.eta, (
                (0, 0), (self.ksize - 1, self.ksize - 1), (self.ksize - 1, self.ksize - 1), (0, 0)),
                             'constant', constant_values=0)

        if self.method == 'SAME':
            pad_eta = np.pad(self.eta, (
                (0, 0), (self.ksize // 2, self.ksize // 2), (self.ksize // 2, self.ksize // 2), (0, 0)),
                             'constant', constant_values=0)

        flip_weights = np.flipud(np.fliplr(self.weights))
        flip_weights = flip_weights.swapaxes(2, 3)
        col_flip_weights = flip_weights.reshape([-1, self.input_channels])
        col_pad_eta = np.array([im2col(pad_eta[i][np.newaxis, :], self.ksize, self.stride)
                                for i in range(self.batchsize)])
        next_eta = np.dot(col_pad_eta, col_flip_weights)
        next_eta = np.reshape(next_eta, self.input_shape)
        return next_eta

    def backward(self, alpha=0.00001, weight_decay=0.0004):
        '''
        :param alpha: 学习率
        :param weight_decay: 衰退率
        更新网络中的权重
        '''
        # self.weights *= (1 - weight_decay)
        # self.bias *= (1 - weight_decay)
        self.weights -= alpha * self.w_gradient
        self.bias -= alpha * self.bias

        self.w_gradient = np.zeros(self.weights.shape)
        self.b_gradient = np.zeros(self.bias.shape)


def im2col(image, ksize, stride):
    '''
    :param image: 输入的图片矩阵
    :param ksize: kernel的长度
    :param stride: 步长
    :return: 转换后的image，便于进行卷积运算
    '''
    image_col = []
    for i in range(0, image.shape[1] - ksize + 1, stride):
        for j in range(0, image.shape[2] - ksize + 1, stride):
            col = image[:, i:i + ksize, j:j + ksize, :].reshape([-1])
            image_col.append(col)
    image_col = np.array(image_col)
    return image_col


def get_output(trains, targets, alpha, epoch, blurry_image):
    '''
    :param trains: 训练集7个模糊图片
    :param targets: 7个高清图片
    :param alpha: 学习率
    :param epoch: 训练次数
    :param blurry_image: 待超分辨重建的图片
    :return: 重建后的高清图片
    '''
    shape = trains[0].shape
    conv1 = Conv2D(shape, 3, 64)
    conv2 = Conv2D(shape, 64, 64)
    conv3 = Conv2D(shape, 64, 64)
    conv4 = Conv2D(shape, 64, 64)
    conv5 = Conv2D(shape, 64, 64)
    conv6 = Conv2D(shape, 64, 64)
    conv7 = Conv2D(shape, 64, 64)
    conv8 = Conv2D(shape, 64, 64)
    final_out = Conv2D(shape, 64, 3)
    relu = Relu()
    mse = MSE()
    losses = []
    # trains = [np.ones((1, 32, 32, 3)) for _ in range(7)]
    # targets = [np.ones((1, 32, 32, 3)) for _ in range(7)]
    for i in range(epoch):
        for j in range(7):
            conv1_out = relu.forward(conv1.forward(trains[j]))
            conv2_out = relu.forward(conv2.forward(conv1_out))
            conv3_out = relu.forward(conv3.forward(conv2_out))
            conv4_out = relu.forward(conv4.forward(conv3_out))
            conv5_out = relu.forward(conv5.forward(conv4_out))
            conv6_out = relu.forward(conv6.forward(conv5_out))
            conv7_out = relu.forward(conv7.forward(conv6_out))
            conv8_out = relu.forward(conv8.forward(conv7_out))
            out = final_out.forward(conv8_out)
            train_loss = mse.cal_loss(out, targets[j])
            losses.append(train_loss)
            conv1.gradient(
                conv2.gradient(
                    conv3.gradient(
                        conv4.gradient(
                            conv5.gradient(
                                conv6.gradient(
                                    conv7.gradient(
                                        conv8.gradient(
                                            final_out.gradient(
                                                mse.gradient(out, targets[j])))
                                    )))))))
            # alpha = alpha*(1-0.000005)**(i*7+j)
            final_out.backward(alpha=alpha)
            conv8.backward(alpha=alpha)
            conv7.backward(alpha=alpha)
            conv6.backward(alpha=alpha)
            conv5.backward(alpha=alpha)
            conv4.backward(alpha=alpha)
            conv3.backward(alpha=alpha)
            conv2.backward(alpha=alpha)
            conv1.backward(alpha=alpha)
            print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) +
                  "  epoch: %d  step: %d loss: %.4f alpha: %.1e" % (i, j, train_loss, alpha))
    conv1_out = relu.forward(conv1.forward(blurry_image))
    conv2_out = relu.forward(conv2.forward(conv1_out))
    conv3_out = relu.forward(conv3.forward(conv2_out))
    conv4_out = relu.forward(conv4.forward(conv3_out))
    conv5_out = relu.forward(conv5.forward(conv4_out))
    conv6_out = relu.forward(conv6.forward(conv5_out))
    conv7_out = relu.forward(conv7.forward(conv6_out))
    conv8_out = relu.forward(conv8.forward(conv7_out))
    sharp_image = final_out.forward(conv8_out)
    plt.plot(losses)
    plt.show()
    return sharp_image


def get_trains_targets(img_name):
    '''
    :param img_name: 待重建的图片名
    :return: [训练集7个模糊图片], [对应的7个高清图片]
    '''
    img = cv2.imread(img_name)
    targets = []  # [(50,50)------(44,44)]
    targets.append(img)
    for i in range(7):
        targets.append(cv2.resize(img, (img.shape[0] - i - 1, img.shape[1] - i - 1)))  # sharp image
    trains = []  # [(50,50)------(44,44)]
    for i in range(7):
        trains.append(
            cv2.resize(targets[i + 1], (targets[i + 1].shape[0] + 1, targets[i + 1].shape[1] + 1)))  # blurry image
    targets = targets[0:-1]
    trains = [train[np.newaxis, :, :, :] for train in trains]
    targets = [target[np.newaxis, :, :, :] for target in targets]
    return trains, targets


def main(img_name, alpha, resize_time, epoch):
    '''
    :param img_name: 待超清重建的图片名
    :param alpha: 学习率
    :param resize_time: 重建次数，重建一次，shape+1
    :param epoch: 每次重建的训练次数
    :return: 输出重建后的图片到output_root
    '''
    output_root = "./output_8_conv"
    trains, targets = get_trains_targets(img_name)
    blurry_image = cv2.resize(targets[0][0], (targets[0][0].shape[0] + 1, targets[0][0].shape[1] + 1))
    blurry_image = blurry_image[np.newaxis, :, :, :]
    sharp_one = get_output(trains, targets, epoch=epoch, blurry_image=blurry_image, alpha=alpha)  # 51, 51
    img_out = sharp_one[0]
    out_name = os.path.join(output_root, os.path.basename(img_name).split('.')[0],
                            str(img_out.shape[0]) + ".jpg")
    if not os.path.exists(os.path.dirname(out_name)):
        os.makedirs(os.path.dirname(out_name))
    cv2.imwrite(out_name, img_out)
    for i in range(resize_time - 1):
        trains = trains[0:-1]
        trains.insert(0, blurry_image)
        targets = targets[0:-1]
        targets.insert(0, sharp_one)
        blurry_image = cv2.resize(targets[0][0], (targets[0][0].shape[0] + 1, targets[0][0].shape[1] + 1))
        blurry_image = blurry_image[np.newaxis, :, :, :]
        sharp_one = get_output(trains, targets, epoch=epoch, blurry_image=blurry_image,
                               alpha=alpha / (i + 1))  # 51+i+1, 51+i+1
        img_out = sharp_one[0]
        out_name = os.path.join(output_root, os.path.basename(img_name).split('.')[0],
                                str(img_out.shape[0]) + ".jpg")
        cv2.imwrite(out_name, img_out)


def t_cnn():
    '''
    测试cnn
    '''
    trains = [np.ones((1, 32, 32, 3)) for _ in range(7)]
    targets = [np.ones((1, 32, 32, 3)) + 1 for i in range(7)]
    alpha = 1e-6
    shape = (1, 32, 32, 3)
    conv1 = Conv2D(shape, 3, 64)
    final_out = Conv2D(shape, 64, 3)
    relu = Relu()
    mse = MSE()
    losses = []
    for i in range(10):
        for j in range(7):
            conv1_out = relu.forward(conv1.forward(trains[j]))
            out = final_out.forward(conv1_out)
            train_loss = mse.cal_loss(out, targets[j])
            losses.append(train_loss)
            conv1.gradient(final_out.gradient(mse.gradient(out, targets[j])))
            final_out.backward(alpha=alpha)
            conv1.backward(alpha=alpha)
            print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) +
                  "  epoch: %d  step: %d loss: %.4f alpha: %.1e" % (i, j, train_loss, alpha))
    plt.plot(losses)
    plt.title("loss - epoch")
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()


if __name__ == "__main__":
    # t_cnn()
    main('./image0.jpg', alpha=3e-11, resize_time=2, epoch=300)
