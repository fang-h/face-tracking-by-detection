import numpy as np
import cv2


class Visualizer(object):

    def __init__(self, config):
        self.config = config
        self.start_collision = -1
        pass

    def draw_detections(self, img, outputs, frame_id):
        if len(outputs) > 0:
            for output in outputs:
                tag = list(output.keys())[0]
                bbox = output[tag]

                x_l = bbox[0]
                x_r = bbox[1]
                y_t = bbox[2]
                y_d = bbox[3]

                width = x_r - x_l + 1
                height = y_d - y_t + 1
                border = height if height >= width else width

                color = np.random.randint(0, 255, (1, 3))[0]
                color = [int(color[0]), int(color[1]), int(color[2])]  # numpy.int转化为python.int
                self.draw_corner(img, x_l, x_r, y_t, y_d, border, color)
                self.draw_bbox(img, x_l, x_r, y_t, y_d, color)
        if self.config.TO_SHOW:
            cv2.imwrite('./detect_result/' + str(frame_id) + '.jpg', img)

    def draw_tracking(self, img, instances, frame_id, show_history_information):
        """Args：
                img: 帧图像
                instances：总控中存在的所有实例
                frame_id: 视频中帧图像的序号
                show_temporal_information:
        """
        if len(instances) > 0:
            for instance in instances:
                if instance.num_of_detect >= self.config.MIN_CONTINUE_DETECTOR:  # ?
                    # 获取instance的最新状态信息latest_history：[tag, bbox[0], bbox[1], bbox[2], bbox[3], color]
                    latest_history = instance.get_latest_record()
                    # 获取人脸信息
                    face = instance.face_id
                    # 获取人脸的表情信息
                    emotion = instance.emotion
                    tag = face + '/' + emotion

                    x_l = latest_history[1]
                    x_r = latest_history[2]
                    y_t = latest_history[3]
                    y_d = latest_history[4]
                    color = latest_history[5]
                    center_x = int((x_l + x_r) / 2)
                    center_y = int((y_t + y_d) / 2)
                    width = x_r - x_l + 1
                    height = y_d - y_t + 1
                    border = height if width >= height else width

                    # 画标签
                    self.draw_tag(img, tag, x_l, x_r, y_t, y_d, border)
                    # 画框
                    self.draw_bbox(img, x_l, x_r, y_t, y_d, color)
                    # 加粗角落线条
                    self.draw_corner(img, x_l, x_r, y_t, y_d, border, color)
                    # 描中心点
                    self.draw_center(img, center_x, center_y, color)
                    # 是否画出帧序号
                    if self.config.SHOW_FRAME_ID:
                        self.draw_frameid(img, str(frame_id))

                    # 如果需要画出该instance在以往时刻的信息(主要是中心点)
                    if show_history_information:
                        history = instance.history
                        for temp in history[:-1]:
                            temp_left = temp[1]
                            temp_right = temp[2]
                            temp_top = temp[3]
                            temp_down = temp[4]
                            temp_cx = int((temp_left + temp_right) / 2)
                            temp_cy = int((temp_top + temp_down) / 2)
                            temp_color = temp[5]
                            self.draw_center(img, temp_cx, temp_cy, temp_color)
        # 保存结果
        if self.config.TO_SHOW:
            cv2.imwrite(self.config.SHOW_PATH + str(frame_id) + '.jpg', img)

    # def drawing_all(self, img, instances, frame_id, is_collision, show_history_information):
    #     if len(instances) > 0:
    #         for instance in instances:
    #             if not show_temporal_information:
    #                 # no need to show historical information (centers)
    #                 ins = instance.get_latest_record()
    #                 # [tag, bbx_left, bbx_right, bbx_up, bbx_bottom, color]
    #                 face = instance.face_id
    #                 emotion = instance.emotion
    #                 tag = face + '/' + emotion
    #
    #                 left = ins[1]
    #                 right = ins[2]
    #                 top = ins[3]
    #                 bottom = ins[4]
    #                 color = ins[5]
    #
    #                 center_x = int((left + right) / 2)
    #                 center_y = int((top + bottom) / 2)
    #
    #                 # get border
    #                 bbox_width = right - left + 1
    #                 bbox_height = bottom - top + 1
    #                 border = bbox_height if bbox_width >= bbox_height else bbox_width
    #
    #                 # draw tag
    #                 self.draw_tag(img, tag, left, right, top, bottom, border)
    #                 # draw bbx
    #                 self.draw_bbox(img, left, right, top, bottom, color)
    #                 # draw corner of the bbx
    #                 self.draw_corner(img, left, right, top, bottom, border, color)
    #                 # draw center
    #                 self.draw_center(img, center_x, center_y, color)
    #                 if self.config.SHOW_FRAME_ID:
    #                     # draw frame id
    #                     self.draw_frameid(img, str(frame_id))
    #                 if is_collision:
    #                     self.start_collision = frame_id
    #                 if self.start_collision != -1:
    #                     if frame_id - self.start_collision <= self.config.SHOW_COLLISION_THRE:  # 50
    #                         self.draw_collision(img)
    #             else:
    #                 # show temporal centers
    #                 history = instance.history
    #                 ins = instance.get_latest_record()
    #
    #                 face = instance.face_id
    #                 emotion = instance.emotion
    #                 tag = face + '/' + emotion
    #                 left = ins[1]
    #                 right = ins[2]
    #                 top = ins[3]
    #                 bottom = ins[4]
    #                 color = ins[5]
    #
    #                 # get border
    #                 bbox_width = right - left + 1
    #                 bbox_height = bottom - top + 1
    #                 border = bbox_height if bbox_width >= bbox_height else bbox_width
    #
    #                 # draw tag
    #                 self.draw_tag(img, tag, left, right, top, bottom, border)
    #                 # draw bbx
    #                 self.draw_bbox(img, left, right, top, bottom, color)
    #                 # draw corner of the bbx
    #                 self.draw_corner(img, left, right, top, bottom, border, color)
    #                 # draw temporal centers
    #                 for temp in history:
    #                     temp_left = temp[1]
    #                     temp_right = temp[2]
    #                     temp_top = temp[3]
    #                     temp_bottom = temp[4]
    #                     temp_cx = int((temp_left + temp_right) / 2)
    #                     temp_cy = int((temp_top + temp_bottom) / 2)
    #                     temp_color = temp[5]
    #                     self.draw_center(img, temp_cx, temp_cy, temp_color)
    #                 if self.config.SHOW_FRAME_ID:
    #                     # draw frame id
    #                     self.draw_frameid(img, str(frame_id))
    #                 if is_collision:
    #                     self.start_collision = frame_id
    #                 if self.start_collision != -1:
    #                     if frame_id - self.start_collision <= self.config.SHOW_COLLISION_THRE:  # 50
    #                         self.draw_collision(img)

    # def showing_tracking_blobs(self, img, blobs, frame_id, show_temporal_information):
    #     img_drawing = img.copy()
    #     if len(blobs) > 0:
    #         for blob in blobs:
    #             if not show_temporal_information:
    #                 # no need to show historical information (centers)
    #                 ins = blob.get_latest_record()
    #                 # [bbx_left, bbx_right, bbx_up, bbx_bottom, color]
    #                 left = ins[0]
    #                 right = ins[1]
    #                 top = ins[2]
    #                 bottom = ins[3]
    #                 color = ins[4]
    #
    #                 center_x = int((left + right) / 2)
    #                 center_y = int((top + bottom) / 2)
    #
    #                 # get border
    #                 bbx_width = right - left + 1
    #                 bbx_height = bottom - top + 1
    #                 border = bbx_height if bbx_width >= bbx_height else bbx_width
    #
    #                 # draw bbx
    #                 self.draw_bbx(img_drawing, left, right, top, bottom, color)
    #                 # draw corner of the bbx
    #                 self.draw_corner(img_drawing, left, right, top, bottom, border, color)
    #                 # draw center
    #                 self.draw_center(img_drawing, center_x, center_y, color)
    #             else:
    #                 # show temporal centers
    #                 history = blob.history
    #                 ins = blob.get_latest_record()
    #
    #                 left = ins[0]
    #                 right = ins[1]
    #                 top = ins[2]
    #                 bottom = ins[3]
    #                 color = ins[4]
    #
    #                 # get border
    #                 bbox_width = right - left + 1
    #                 bbox_height = bottom - top + 1
    #                 border = bbox_height if bbox_width >= bbox_height else bbox_width
    #
    #                 # draw bbx
    #                 self.draw_bbox(img_drawing, left, right, top, bottom, color)
    #                 # draw corner of the bbx
    #                 self.draw_corner(img_drawing, left, right, top, bottom, border, color)
    #                 # draw temporal centers
    #                 for temp in history:
    #                     temp_left = temp[0]
    #                     temp_right = temp[1]
    #                     temp_top = temp[2]
    #                     temp_bottom = temp[3]
    #                     temp_cx = int((temp_left + temp_right) / 2)
    #                     temp_cy = int((temp_top + temp_bottom) / 2)
    #                     temp_color = temp[4]
    #                     self.draw_center(img_drawing, temp_cx, temp_cy, temp_color)
    #                 if self.config.SHOW_FRAME_ID:
    #                     # draw frame id
    #                     self.draw_frameid(img, str(frame_id))
    #     return img_drawing

    def draw_corner(self, img, x_l, x_r, y_t, y_d, border, color):
        """用于加粗bbox的四个角"""
        length = 2 if border / 5 <= 1 else border / 5
        x_l_end = int(x_l + length - 1)
        x_r_end = int(x_r - length + 1)
        y_t_end = int(y_t + length - 1)
        y_d_end = int(y_d - length + 1)
        line_color = (color[0], color[1], color[2])

        # 左上角
        cv2.line(img, (x_l, y_t), (x_l_end, y_t), line_color, 2, lineType=8)
        cv2.line(img, (x_l, y_t), (x_l, y_t_end), line_color, 2, 8)

        # 右上角
        cv2.line(img, (x_r, y_t), (x_r_end, y_t), line_color, 2, 8)
        cv2.line(img, (x_r, y_t), (x_r, y_t_end), line_color, 2, 8)

        # 左下角
        cv2.line(img, (x_l, y_d), (x_l_end, y_d), line_color, 2, 8)
        cv2.line(img, (x_l, y_d), (x_l, y_d_end), line_color, 2, 8)

        # 右下角
        cv2.line(img, (x_r, y_d), (x_r_end, y_d), line_color, 2, 8)
        cv2.line(img, (x_r, y_d), (x_r, y_d_end), line_color, 2, 8)

    def draw_bbox(self, img, x_l, x_r, y_t, y_d, color):
        cv2.rectangle(img, (x_l, y_t), (x_r, y_d), color, 1)

    def draw_tag(self, img, tag, x_l, x_r, y_t, y_d, border):
        text_size = cv2.getTextSize(tag, cv2.FONT_HERSHEY_DUPLEX, 0.45, 1)
        text_width = text_size[0][0]
        text_height = text_size[0][1]
        text_baseline = text_size[1]
        text_org = (x_l, y_t - text_baseline)
        text_bbox_left_top = (x_l, y_t - int(1.5 * text_baseline) - text_height)
        text_bbox_right_down = (x_l + text_width, y_t)

        # 背景
        cv2.rectangle(img, text_bbox_left_top, text_bbox_right_down, (145, 145, 145), cv2.FILLED)

        # 标签
        cv2.putText(img, tag, text_org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), True)

    def draw_center(self, img, cx, cy, color):
        cv2.circle(img, (cx, cy), 2, color, cv2.FILLED)

    def draw_frameid(self, img, id_str):
        cv2.putText(img, id_str, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), True)





