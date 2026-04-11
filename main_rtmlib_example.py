
# import torch
# import cv2
# from rtmlib import Body, Custom, PoseTracker, Wholebody, Wholebody3d, Hand, draw_skeleton

# cap = cv2.VideoCapture(0)  # for video file instead of webcam, use cap = cv2.VideoCapture('./demo.mp4')

# device = 'cuda'
# backend = 'onnxruntime'
# openpose_skeleton = False
# print(torch.__version__)

# # detector = PoseTracker(Body,
# #                          mode='performance',
# #                          det_frequency=20,  # detect every 10 frames
# #                          backend=backend, device=device,
# #                          tracking=False,
# #                          to_openpose=False)

# # detector = Hand(to_openpose=openpose_skeleton,
# #              backend=backend,
# #              device=device)

# # Example: Hand in lightweight mode
# # detector = Custom(to_openpose=openpose_skeleton,
# #                 det_class='RTMDet',
# #                 det='https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmdet_nano_8xb32-300e_hand-267f9c8f.zip',
# #                 det_input_size=(320,320),
# #                 pose_class='RTMPose',
# #                 pose='https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.zip',
# #                 pose_input_size=(256, 256),
# #                 backend=backend,
# #                 device=device)

# # detector = Wholebody(to_openpose=openpose_skeleton,
# #                       backend=backend,
# #                       device=device)

# detector = PoseTracker(
#     Wholebody3d,
#     det_frequency=21,
#     tracking=False,
#     to_openpose=openpose_skeleton,
#     backend=backend,
#     device=device)


# # # Or with a custom class
# # from functools import partial
# # custom = partial(Custom,
# #                 det_class='YOLOX',
# #                 det='https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip',
# #                 det_input_size=(640, 640),
# #                 pose_class='RTMPose',
# #                 pose='https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-body7_pt-body7-halpe26_700e-256x192-4d3e73dd_20230605.zip',
# #                 pose_input_size=(192, 256))
# # pose_tracker = PoseTracker(custom,
# #                         det_frequency=10,
# #                         backend=backend, device=device,
# #                         to_openpose=False)
# frame_idx = 0
# while cap.isOpened():
#     success, frame = cap.read()
#     frame_idx += 1
#     if not success:
#         break

#     #keypoints, scores = detector(frame)
#     keypoints, scores, keypoints_simcc, keypoints_2d = detector(frame)


#     img_show = frame.copy()
#     img_show = draw_skeleton(img_show,
#                              keypoints_2d,
#                              scores,
#                              openpose_skeleton=openpose_skeleton,
#                              kpt_thr=0.43)
    
#     img_show = cv2.resize(img_show, (960, 540))
#     cv2.imshow('img', img_show)
#     cv2.waitKey(10)