import os
import time
import config
import argparse

from video_helper import VideoHelper
from multiple_object_controller import MultipleObjectController
from face_detector import face_detector_model, face_detector
from visualizer import Visualizer
import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='main')
parser.add_argument('--model_name', default='HRFaceBoxes', type=str,
                    help='select different face detect algorithm')
args = parser.parse_args()


def main():

    video_helper = VideoHelper(config)
    object_controller = MultipleObjectController(video_helper, config)
    visualizer = Visualizer(config)
    model, default_box = face_detector_model(model_name=args.model_name)

    cur_frame_counter = 0
    detection_loop_counter = 0

    while video_helper.not_finished(cur_frame_counter):
        # frame是原始的帧图像，frame_show是用于显示检测结果的图像
        frame, frame_show = video_helper.get_frame()

        # 情况1：每帧都进行检测
        if config.NUM_JUMP_FRAMES == 0:
            # face_detector检测人脸，返回的detections：[{'face':[x_l, x_r, y_t, y_d]},{...}]
            detections = face_detector(frame, cur_frame_counter, model, default_box)

            # 采用object_controller类的update_with_detections进行更新
            object_controller.update_with_detections(detections, frame, cur_frame_counter)
        # 情况二：跳帧检测
        else:
            # 需要检测的帧
            if detection_loop_counter % config.NUM_JUMP_FRAMES == 0:
                detection_loop_counter = 0
                detections = face_detector(frame, cur_frame_counter, model, default_box)
                # 更新的时候结合detections
                object_controller.update_with_detections(detections, frame, cur_frame_counter)
            # 不需要检测的帧
            else:
                # 直接由跟踪算法进行预测更新
                object_controller.update_without_detections(frame, cur_frame_counter)
        # 利用visualizer类进行展示
        show_history_information = False
        visualizer.draw_tracking(frame_show, object_controller.instances, cur_frame_counter,
                                 show_history_information)

        # if cur_frame_counter == 90 or cur_frame_counter == 93:
        #     visualizer.draw_detections(frame, detections, cur_frame_counter)

        cur_frame_counter += 1
        detection_loop_counter += 1

    video_helper.end()


if __name__ == "__main__":
    main()