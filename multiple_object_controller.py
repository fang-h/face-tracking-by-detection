from instance import Instance
import utils


class MultipleObjectController(object):
    """MutilpleObjectController类用来处理所有实例的追踪过程，是一个总控"""

    def __init__(self, video_helper, config):
        self.instances = []
        self.instances_blob = []
        self.video_helper = video_helper
        self.config = config

    def update_with_detections(self, detections, frame, frame_id):
        """采用detections和predictions对跟踪系统进行更新
        :param：detections：[{'tag1', [x_l, x_r, y_t, y_d]}, {'tag2', [x_l, x_r, y_t, y_d]},...]
        :param: frame:帧图像
        :param：frame_id:视频中当前frame的序号"""

        """情况一：总控中不存在instance"""
        if len(self.instances) == 0:
            for det in detections:
                # 创建一个instance用来记录该det
                instance = Instance(self.config, self.video_helper, frame)
                tag = list(det.keys())[0]
                bbox = det[tag]
                instance.add_to_track_with_correction(tag, bbox, frame)
                # 将该instance加入总控的instances列表中
                self.instances.append(instance)
            return True

        """情况二：总控中存在instance"""

        # 计算iou
        det_track_iou = {}  # 记录检测到的bbox和每一个跟踪的bbox的iou值
        track_det_iou = {}  # 记录跟踪的bbox和每一个检测到的bbox的iou值
        # 对总控中的每一个instance
        for i, instance in enumerate(self.instances):
            if not track_det_iou.__contains__(i):
                track_det_iou[i] = []
            # 由跟踪算法预测bbox：[x_l, x_r, y_t, y_d]
            predict_bbox = instance.get_predict_bbox(frame)
            # 对检测到的每一个bbox
            for j, det in enumerate(detections):
                if not det_track_iou.__contains__(j):
                    det_track_iou[j] = []
                detect_bbox = list(det.values())[0]
                iou = utils.get_iou(predict_bbox, detect_bbox)
                track_det_iou[i].append([j, iou])  # 表示第i个instance的预测bbox与第j个检测到的bbox的iou值
                det_track_iou[j].append([i, iou])  # 表示第j个检测到的bbox与第i个instance的预测bbox的iou值
        # if frame_id == 90:
        #     print(track_det_iou)
        #     print(track_det_iou)

        # 将instance都设置为未匹配状态
        for instance in self.instances:
            instance.has_match = False

        # 开始进行匹配，
        assigned_instances = []  # 记录被匹配的instance
        assigned_detections = []  # 记录被匹配的detections
        # 对第i个instance
        for i, id_iou in track_det_iou.items():  # id_iou:[[id0, iou0],[id1, iou1], ...]
            # 找到该instance对应的iou值最大的detect_bbox，id为match_det_id
            matched_det_id = utils.get_max_iou_id(id_iou)
            if matched_det_id is not None:
                # 找到id为match_det_id的detect_bbox对应的iou值最大的predict_bbox
                matched_track_id = utils.get_max_iou_id(det_track_iou[matched_det_id])
                # 如果双向匹配成功, 而且iou大于阈值
                if matched_track_id == i and \
                        id_iou[matched_det_id][1] > self.config.DETECTION_AND_INSTANCE_IDENTICAL_IOU_THRESHOLD:
                # if matched_track_id == i:
                    # assigned_instances和assigned_detections一一对应
                    assigned_instances.append(matched_track_id)
                    assigned_detections.append(matched_det_id)

        # 开始修正匹配好的instance和detections
        assigned_detection_id = []
        if assigned_detections is not None and assigned_instances is not None:
            # 对每一个匹配的instance
            for idx, instance_id in enumerate(assigned_instances):
                # 找到该instance对应的匹配detection的id
                detection_id = assigned_detections[idx]
                assigned_detection_id.append(detection_id)
                # 改变该instance的匹配状态为True
                self.instances[instance_id].has_match = True
                # 进行更新
                tag = list(detections[detection_id].keys())[0]
                bbox = detections[detection_id][tag]
                self.instances[instance_id].add_to_track_with_correction(tag, bbox, frame)
                # 一旦在某个时刻某个instance被检测到，就将该instance的num_misses的值设置为0
                self.instances[instance_id].num_misses = 0

        for instance in self.instances:
            # 对于未被匹配的instance
            if instance.has_match is False:
                # 将未匹配的instance的num_misses的值+1，表示该时刻该instance未被检测到
                instance.num_misses += 1

        # 可能存在已经不在的instance，去掉这些instance
        self.remove_dead_instances()

        # 对于未匹配的detections，创建新的instance
        unassigned_detection_id = list(set(range(0, len(detections))) - set(assigned_detection_id))
        for idx in range(0, len(detections)):
            if idx in unassigned_detection_id:
                tag = list(detections[idx].keys())[0]
                bbox = detections[idx][tag]
                # 对于检测到而且为匹配的bbox，首先判断该bbox是否是一个好的detection
                # 如果和现有的instance的bbox的ios值大于设定的阈值则表明不是一个好的detection
                if self.is_good_detection(bbox):
                    instance = Instance(self.config, self.video_helper, frame)
                    instance.add_to_track_with_correction(tag, bbox, frame)
                    self.instances.append(instance)

    def update_without_detections(self, frame, frame_id):
        """直接采用跟踪算法预测进行更新"""
        # 对于self.instances中的每一个instance
        for instance in self.instances:
            bbox = instance.get_predict_bbox(frame)
            # 获取该instance上一个状态(history:[tag, bbox[0], bbox[1], bbox[2], bbox[3], color])
            tag = instance.get_latest_record()[0]
            instance.add_to_track_without_correction(tag, bbox, frame)
            instance.num_misses += 1
        self.remove_dead_instances()

    # def update_still(self, frame, frame_id):
    #     for instance in self.instances:
    #         bbox = instance.get_latest_bbox()
    #         tag = instance.get_latest_record()[0]
    #         instance.add_to_track_without_correction(tag, bbox, frame)
    #         instance.num_misses += 1
    #     self.remove_dead_instance()

    def remove_dead_instances(self):
        self.delete_duplicate_tracks()
        self.delete_still_tracks()
        self.delete_singular_tracks()

        self.instances = [instance for instance in self.instances if
                          (instance.num_misses < self.config.MAX_NUM_MISSING_PERMISSION
                           and instance.delete_still is False
                           and instance.delete_duplicate is False
                           and instance.delete_singular is False)]

    def delete_duplicate_tracks(self):
        """删除instances中某些instance之间的bbox的iou阈值太高的instance"""
        for i in range(len(self.instances)):
            ins1 = self.instances[i]
            for j in range(len(self.instances)):
                if i == j:
                    continue
                ins2 = self.instances[j]
                # 如果两个不同的instance的bbox的iou值大于设定的阈值
                if utils.check_instance_identical_by_iou(ins1,
                                                         ins2, self.config.INSTANCE_IDENTICAL_IOU_THRESHOLD):
                    # 将拥有更多history记录(更早被detect)的instance留下
                    if ins1.get_age() > ins2.get_age():
                        ins2.delete_duplicate = True
                        ins1.delete_duplicate = False
                    else:
                        ins1.delete_duplicate = True
                        ins2.delete_duplicate = False

    # ？
    def delete_still_tracks(self):
        for instance in self.instances:
            if len(instance.history) > self.config.NUM_DELETE_STILL:
                still_counter = 0
                for i in range(1, self.config.NUM_DELETE_STILL + 1):
                    bbox_i = instance.get_ith_bbox(-i)
                    bbox_i_1 = instance.get_ith_bbox(-i - 1)
                    # 统计
                    sum_still = utils.get_sum_still(bbox_i, bbox_i_1)
                    if sum_still == 0:
                        still_counter += 1
                if still_counter >= self.config.NUM_DELETE_STILL:
                    instance.delete_still = True
                else:
                    instance.delete_still = False

    def delete_singular_tracks(self):
        # 去除掉一些面积太大或者太小，长宽比太大的instance
        for instance in self.instances:
            curr_bbox = instance.get_latest_bbox()
            x_l = curr_bbox[0]
            x_r = curr_bbox[1]
            y_t = curr_bbox[2]
            y_d = curr_bbox[3]
            area = utils.get_area_from_coord(x_l, x_r, y_t, y_d)
            wh_ratio = utils.get_wh_ratio_from_coord(x_l, x_r, y_t, y_d)
            if area > self.config.AREA_MAXIMUM:
                instance.delete_singular = True
            elif area < self.config.AREA_MINIMUM:
                instance.delete_singular = True
            elif wh_ratio > self.config.WH_RATIO_THRESHOLD:
                instance.delete_singular = True
            else:
                instance.delete_singular = False

    def is_good_detection(self, bbox):
        # 判断检测到而未匹配的bbox是否是一个好的detection
        # 判断的依据是计算该bbox和所有已经存在的instance的bbox的交集部分的面积与该instance的bbox的面积的比值ios是否大于设定的阈值
        for instance in self.instances:
            if utils.check_bbox_identical_by_ios(instance.get_latest_bbox(), bbox,
                                                 self.config.BBOX_IDENTICAL_IOS_THRESHOLD):
                # 大于设定的阈值则表明不是一个好的detection
                return False
        return True
