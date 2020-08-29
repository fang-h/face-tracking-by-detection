import cv2
import numpy as np


class VideoHelper(object):

    def __init__(self, config):
        self.video_in = cv2.VideoCapture()
        self.video_in.open(config.VIDEO_NAME)

        self.frame_width = int(self.video_in.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.video_in.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_fps = int(self.video_in.get(cv2.CAP_PROP_FPS))

        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        self.video_out = cv2.VideoWriter(config.VIDEO_SAVING_NAME, fourcc, self.frame_fps,
                                         (self.frame_width, self.frame_height))
        self.video_blob_out = cv2.VideoWriter(config.VIDEO_SAVING_BLOB_NAME, fourcc, self.frame_fps,
                                              (config.BACK_RESIZE_WIDTH, config.BACK_RESIZE_HEIGHT))
        self.finish_frame_nums = config.FINISH_CUT_FRAME

    def not_finished(self, cut_frame):
        if self.video_in.isOpened():
            if self.finish_frame_nums == 0:
                return True
            if cut_frame < self.finish_frame_nums:
                return True
            else:
                return False
        else:
            print('Video is not opened')
            return False

    def get_frame(self):
        ret, frame = self.video_in.read()
        if ret is False:
            print('Video is done')
            exit()
        frame_show = frame.copy()
        return frame, frame_show

    def write_video(self, image):
        self.video_out.write(image)

    def end(self):
        self.video_in.release()
        self.video_out.release()
        self.video_blob_out.release()
