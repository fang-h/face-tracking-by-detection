"""手动实现Kalman Filter, 不采用Opencv里面封装的KalmanFilter类"""


import numpy as np


class KalmanTracker(object):

    def __init__(self, dynamParams, measureParams, controlParams=None):
        """Args:
                dynamParams: 状态量数量
                measureParams: 观测量数量
                controlParams: 外部控制量数量"""

        self.statePre = np.zeros((dynamParams, 1), np.float32)  # 公式1的结果
        self.statePost = np.zeros((dynamParams, 1), np.float32)  # 公式4的结果
        self.transitionMatrix = np.zeros((dynamParams, dynamParams), np.float32)  # 公式1中的状态转移矩阵
        if controlParams is not None:
            self.controlMatrix = np.zeros((dynamParams, controlParams), np.float32)  # 公式1中的控制矩阵
        else:
            self.controlMatrix = None
        self.measurementMatrix = np.zeros((measureParams, dynamParams), np.float32)  # 公式2中的测量矩阵
        self.processNoiseCov = np.zeros((dynamParams, dynamParams), np.float32)  # 公式2中的系统噪声协方差矩阵
        self.measurementNoiseCov = np.zeros((measureParams, measureParams), np.float32)  # 公式3中的测量噪声协方差
        self.errorCovPre = np.zeros((dynamParams, dynamParams), np.float32)  # 公式2中的结果(协方差）
        self.gain = np.zeros((dynamParams, measureParams), np.float32)  # 公式3的结果(滤波器增益）
        self.errorCovPost = np.zeros((dynamParams, dynamParams), np.float32)  # 公式5的结果

    def predict(self, control=None):
        # 更新
        self.statePre = np.dot(self.transitionMatrix, self.statePost)  # 公式1
        # 外部控制
        if control is not None:
            self.statePre += np.dot(self.controlMatrix, control)  # 公式1
        # 更新系统协方差矩阵,公式2
        # self.errorCovPre = self.transitionMatrix * self.errorCovPost * self.transitionMatrix.T + self.processNoiseCov
        temp1 = np.dot(self.transitionMatrix, self.errorCovPost)
        temp2 = np.dot(temp1, self.transitionMatrix.T)
        self.errorCovPre = temp2 + self.processNoiseCov

        # 只在进行预测时,Pre和Post进行更新
        self.statePost = self.statePre
        self.errorCovPost = self.errorCovPre
        return self.statePost

    def correct(self, measurement):
        # 计算卡尔曼增益,公式3
        # self.gain = self.errorCovPre * self.measurementMatrix.T * np.linalg.inv(
        #     self.measurementMatrix * self.errorCovPre * self.measurementMatrix.T + self.measurementNoiseCov)
        temp3 = np.dot(self.errorCovPre, self.measurementMatrix.T)
        temp4 = np.dot(self.measurementMatrix, self.errorCovPre)
        temp5 = np.dot(temp4, self.measurementMatrix.T) + self.measurementNoiseCov
        self.gain = np.dot(temp3, np.linalg.inv(temp5))

        # 计算最优估计,公式4
        # self.statePost = self.statePre + self.gain * (measurement - self.measurementMatrix * self.statePre)
        temp6 = np.dot(self.measurementMatrix, self.statePre)
        self.statePost = self.statePre + np.dot(self.gain, (measurement - temp6))

        # 更新系统协方差矩阵
        # self.errorCovPost = self.errorCovPre - self.gain * self.measurementMatrix * self.errorCovPre
        temp7 = np.dot(self.gain, self.measurementMatrix)
        self.errorCovPost = self.errorCovPre - np.dot(temp7, self.errorCovPre)

        # Pre和Post进行更新
        self.statePre = self.statePost
        self.errorCovPre = self.errorCovPost

        return self.statePost

