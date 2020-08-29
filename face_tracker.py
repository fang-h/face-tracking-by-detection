import cv2
import numpy as np
import math

from tracker.kcf_tracker import KCFTracker
from tracker.kalman_tracker import KalmanTracker


class KcfFilter(object):
    def __init__(self, video_helper, frame):
        self.first_run = True
        self.dynamParamsSize = 6
        self.measureParamsSize = 4
        self.kcf = KCFTracker(True, True, True)

    def correct(self, bbox, frame):
        w = bbox[1] - bbox[0] + 1
        h = bbox[3] - bbox[2] + 1
        cx = int(bbox[0] + w / 2)
        cy = int(bbox[2] + h / 2)
        measurement = np.array([[cx, cy, w, h]], dtype=np.float32).T
        if self.first_run is True:
            self.kcf.init(measurement, frame)
            # self.first_run = False   # 每次检测都必须重新初始化， 否则检测没有意义, 这是与kalman不一样的地方
        corrected_res = self.kcf.update(frame)
        # self.velocity = np.array([corrected_res[2], corrected_res[3]])
        corrected_bbox = self.point_form(corrected_res)
        return corrected_bbox

    def get_predict_bbox(self, frame):
        predicted_res = self.kcf.update(frame)
        predicted_bbox = self.point_form(predicted_res)
        return predicted_bbox

    def point_form(self, center_form_bbox):
        cx = center_form_bbox[0]
        cy = center_form_bbox[1]
        w = center_form_bbox[2]
        h = center_form_bbox[3]
        x_l = math.ceil(cx - w / 2.0)
        x_r = math.ceil(cx + w / 2.0)
        y_t = math.ceil(cy - h / 2.0)
        y_d = math.ceil(cy + h / 2.0)
        return [x_l, x_r, y_t, y_d]


class KalmanFilter(object):

    def __init__(self, video_helper, from_cv2=True):
        """Args:
                video_helper:处理视频的类
                from_cv2:是否使用cv2中封装的KalmanFilter"""
        self.dynamParamsSize = 6  # 状态量数量：[cx, cy, vx, vy, w, h]
        self.measureParamsSize = 4  # 观察量数量：[cx, cy, w, h]
        if from_cv2:
            self.kalman = cv2.KalmanFilter(dynamParams=self.dynamParamsSize,
                                           measureParams=self.measureParamsSize)
        else:
            self.kalman = KalmanTracker(dynamParams=self.dynamParamsSize,
                                        measureParams=self.measureParamsSize)
        self.first_run = True
        dT = 1.0 / video_helper.frame_fps

        # 状态转移矩阵
        self.kalman.transitionMatrix = np.array([[1, 0, dT, 0, 0, 0],
                                                [0, 1, 0, dT, 0, 0],
                                                [0, 0, 1, 0, 0, 0],
                                                [0, 0, 0, 1, 0, 0],
                                                [0, 0, 0, 0, 1, 0],
                                                [0, 0, 0, 0, 0, 1]], np.float32)
        # 系统测量矩阵
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0, 0, 0],
                                                 [0, 1, 0, 0, 0, 0],
                                                 [0, 0, 0, 0, 1, 0],
                                                 [0, 0, 0, 0, 0, 1]], np.float32)

        # 系统噪声协方差矩阵
        self.kalman.processNoiseCov = np.array([[0.01, 0, 0, 0, 0, 0],
                                               [0, 0.01, 0, 0, 0, 0],
                                               [0, 0, 1, 0, 0, 0],
                                               [0, 0, 0, 1, 0, 0],
                                               [0, 0, 0, 0, 0.01, 0],
                                               [0, 0, 0, 0, 0, 0.01]], np.float32)

        # 测量噪声协方差
        self.kalman.measurementNoiseCov = np.array([[0.01, 0, 0, 0],
                                                   [0, 0.01, 0, 0],
                                                   [0, 0, 0.01, 0],
                                                   [0, 0, 0, 0.01]], np.float32)

    def get_predict_bbox(self, frame):
        statePre = self.kalman.predict().T[0]
        bbox = self.point_form([statePre[0], statePre[1], statePre[4], statePre[5]])
        return bbox

    def correct(self, bbox, frame):
        # bbox为观测量,bbox: [x_l, x_r, y_t, y_d]
        cx = int((bbox[0] + bbox[1]) / 2.0)
        cy = int((bbox[2] + bbox[3]) / 2.0)
        w = bbox[1] - bbox[0] + 1
        h = bbox[3] - bbox[2] + 1
        measurement = np.array([[cx, cy, w, h]], np.float32).T

        # 系统开始第一次，初始化状态矩阵
        if self.first_run:
            self.kalman.statePre = np.array([[measurement[0], measurement[1],
                                              0, 0, measurement[2], measurement[3]]], np.float32).T

            self.first_run = False  # 注释掉表示每次检测都初始化statePre
        statePost = self.kalman.correct(measurement).T[0]
        correct_bbox = self.point_form([statePost[0], statePost[1], statePost[4], statePost[5]])
        return correct_bbox

    def point_form(self, center_form_bbox):
        cx = center_form_bbox[0]
        cy = center_form_bbox[1]
        w = center_form_bbox[2]
        h = center_form_bbox[3]
        x_l = math.ceil(cx - w / 2.0)
        x_r = math.ceil(cx + w / 2.0)
        y_t = math.ceil(cy - h / 2.0)
        y_d = math.ceil(cy + h / 2.0)
        return [x_l, x_r, y_t, y_d]












