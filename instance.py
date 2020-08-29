import numpy as np

from face_tracker import KalmanFilter
from face_tracker import KcfFilter


class Instance(object):
    """该Instance类用来描述一个追踪的实例，比如一个人脸, 主要包含该实例在不同时刻的状态，以及加入新状态和删除旧状态等操作"""

    def __init__(self, config, video_helper, frame):
        self.config = config

        self.history = []  # 记录不同时刻该实例的一些状态值：history1:[tag, bbox[0]. bbox[1], bbox[2], bbox[3], color]
        self.history_size = config.HISTORY_SIZE = 10  # 允许记录时刻的上限值
        self.face_id = 'None'
        self.history_face_id = 'None'
        self.emotion = 'None'
        self.history_emotion = 'None'

        self.num_misses = 0  # 用于记录instance被漏检的次数
        self.max_misses = config.MAX_NUM_MISSING_PERMISSION
        self.has_match = False
        self.delete_duplicate = False
        self.delete_singular = False
        self.delete_still = False
        self.num_of_detect = 0  # 用于记录该instance被检测到的次数

        # 使用kcf跟踪算法
        if config.KCF:
            self.tracker = KcfFilter(video_helper, frame)
        # 使用kalman跟踪算法
        elif config.KALMAN:
            self.tracker = KalmanFilter(video_helper, config.FROM_CV2)

        # 标注颜色
        color = np.random.randint(0, 255, size=(1, 3))[0]  # numpy.ndarray
        self.color = [int(color[0]), int(color[1]), int(color[2])]  # 转化为python int而不是numpy int
        self.center_color = []
        self.center_color.append(self.color)
        # self.COLOR_FADING_PARAM = config.COLOR_FADING_PARAM
        # self.speed = 0
        # self.direction = 0
        # self.still_history = 0
        # self.predict_next_bbox = None
        # self.num_established = 1

    def add_to_track_with_correction(self, tag, bbox, frame):
        """tag为该框的标志---'face', bbox为检测到的框, frame为当前帧图像"""

        # 通过跟踪算法获得修正后的correct_bbox
        correct_bbox = self.tracker.correct(bbox, frame)
        # 记录一些状态值
        new_history = [tag, correct_bbox[0], correct_bbox[1], correct_bbox[2], correct_bbox[3],
                       self.color]

        # 将new_history加入self.history时分为两种情况
        # 情况一：在于检测到但是未匹配的的bbox所创建的instance，直接加入self.history
        if len(self.history) == 0:
            self.history.append(new_history)
        # 情况二：对于检测到但是匹配的bbox，依次改变self.history中的每一个history的color值，然后加入new_history
        else:
            # for i in range(len(self.history)):
            #     for j in range(3):
            #         changed_color = int((self.config.COLOR_FADING_PARAM - 1) / self.config.COLOR_FADING_PARAM *
            #                             self.history[i][5][j])
            #         changed_color = int(self.history[i][5][j] * 1)
            #         if changed_color < 0:
            #             changed_color = 0
            #         self.history[i][5][j] = changed_color
            self.history.insert(0, new_history)
        # 当self.history记录超过设定值时，删除最先记录的history
        if len(self.history) == self.history_size:
            del self.history[-1]

        # 将instance的num_of_detect加1表示被检测到的次数增加1次
        self.num_of_detect += 1  # ？

    def add_to_track_without_correction(self, tag, bbox, frame):
        """tag为该框的标志---'face', bbox为预测的框, frame为当前帧图像"""
        new_history = [tag, bbox[0], bbox[1], bbox[2], bbox[3],
                       self.color]
        # 直接改变self.history中history的color值，并加入new_history
        # for i in range(len(self.history)):
        #     for j in range(3):
        #         changed_color = int((self.config.COLOR_FADING_PARAM - 1) / self.config.COLOR_FADING_PARAM *
        #                             self.history[i][5][j])
        #         changed_color = int(self.history[i][5][j] * 1)
        #         if changed_color < 0:
        #             changed_color = 0
        #         self.history[i][5][j] = changed_color
        self.history.insert(0, new_history)
        if len(self.history) == self.history_size:
            del self.history[-1]

        # self.num_of_detect += 1  # ?

    def get_predict_bbox(self, frame):
        # 从tracker获取一个预测值
        return self.tracker.get_predict_bbox(frame)

    def get_latest_bbox(self):
        # 从history中获取最后加入的bbox
        if len(self.history) == 0:
            return []
        bbox = self.history[0][1:5]
        return bbox

    def get_ith_bbox(self, i):
        # 从history中获取第i个bbox
        if (0 < i < len(self.history)) or (i < 0 and -i <= len(self.history)):
            return self.history[i][1:5]
        else:
            return []

    def get_first_bbox(self):
        # 获取history中最先加入的bbox
        return self.get_ith_bbox(-1)

    def get_latest_record(self):
        if len(self.history) > 0:
            return self.history[0]
        else:
            return []

    def get_age(self):
        # 返回单当前history中的记录数量
        return len(self.history)





