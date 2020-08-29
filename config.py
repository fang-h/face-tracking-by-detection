"""整个系统的参数配置"""

# 文件读取与保存地址
VIDEO_NAME = 'demo5.mp4'
SHOW_PATH = 'vis/'
VIDEO_SAVING_NAME = 'save_video'
VIDEO_SAVING_BLOB_NAME = 'VID_SAVING_BLOB_NAME'

# 选择
KCF = False  # 是否选择KCF跟踪算法
KALMAN = bool(1 - KCF)  # 是否选择Kalman跟踪算法
FROM_CV2 = True  # Kalman跟踪是否选用cv2封装的
SHOW_FRAME_ID = False  # 是否在图片上显示当前frame在video中的序号
TO_SHOW = True  # 是否显示跟踪结果

# 阈值
DETECTION_AND_INSTANCE_IDENTICAL_IOU_THRESHOLD = 0.35  # 在进行双向匹配的时候,detection和instance的iou
INSTANCE_IDENTICAL_IOU_THRESHOLD = 0.3  # 两个instance之间的bbox的iou阈值,用于判断系统中是否存在多余的instance
BBOX_IDENTICAL_IOS_THRESHOLD = 0.5  # 对于未匹配上的detections,判断其是否是一个好的detections
AREA_MAXIMUM = 10000000  # 检测中允许的最大框面积
AREA_MINIMUM = 400  # 检测中允许的最小框面积
WH_RATIO_THRESHOLD = 3   # 检测中允许的最大边与最小边的比值


# 其他设置
MIN_CONTINUE_DETECTOR = 2  # 在显示时,只显示被检测到的次数大于该参数配置的instance
HISTORY_SIZE = 20  # instance允许记录的最大个数
NUM_JUMP_FRAMES = 3  # 跳帧检测
FINISH_CUT_FRAME = 0  # 最多运行的帧数
NUM_DELETE_STILL = NUM_JUMP_FRAMES + 1  # instance允许连续多个一样的history(bbox)
MAX_NUM_MISSING_PERMISSION = NUM_JUMP_FRAMES + 1  # 允许漏检的最多次数(只更新不修正时也算漏检一次)
COLOR_FADING_PARAM = HISTORY_SIZE * 1000  # 控制instance中每个记录的颜色

# video_out相关参数
BACK_RESIZE_BORDER = 480
BACK_RESIZE_HEIGHT = 240
BACK_RESIZE_WIDTH = 320






