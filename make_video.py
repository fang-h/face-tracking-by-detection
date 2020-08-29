import cv2
import glob
import os
from datetime import datetime


def frames_to_video(fps, save_path, frames_path):
    max_index = len(os.listdir(frames_path))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    imgs = glob.glob(frames_path + "/*.jpg")
    shape_i = cv2.imread(imgs[0]).shape
    videoWriter = cv2.VideoWriter(save_path, fourcc, fps, (shape_i[1], shape_i[0]))

    frames_num = len(imgs)
    for i in range(max_index):
        if os.path.isfile("%s/%d.jpg" % (frames_path, i)):
            print(i)
            frame = cv2.imread("%s/%d.jpg" % (frames_path, i))
            videoWriter.write(frame)
    videoWriter.release()
    return


if __name__ == '__main__':
    t1 = datetime.now()
    video_in = cv2.VideoCapture()
    video_in.open('demo5.mp4')
    fps = int(video_in.get(cv2.CAP_PROP_FPS))
    video_in.release()
    frames_to_video(fps, "video5.mp4", 'vis/')
    t2 = datetime.now()
    print("Time cost = ", (t2 - t1))
    print("SUCCEED !!!")



