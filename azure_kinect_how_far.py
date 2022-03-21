#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import time
import cv2
import copy
import open3d as o3d
import numpy as np
import argparse
import onnxruntime

from yolox.data.data_augment import preproc as preprocess
from yolox.data.datasets import COCO_CLASSES
from yolox.utils import mkdir, multiclass_nms, demo_postprocess


def get_parser():
    parser = argparse.ArgumentParser(description='onnxruntime inference on Azure kinect.')
    parser.add_argument('--config', type=str, default='./config/default_config.json', help='input json kinect config')
    parser.add_argument('--list',
                        action='store_true',
                        help='list available azure kinect sensors')
    parser.add_argument('--device',
                        type=int,
                        default=0,
                        help='input kinect device id')
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="./checkpoints/yolox_s.onnx",
        help="Input your onnx model.",
    )
    parser.add_argument(
        "-s",
        "--score_thr",
        type=float,
        default=0.3,
        help="Score threshould to filter the result.",
    )
    parser.add_argument(
        "--input_shape",
        type=str,
        default="640,640",
        help="Specify an input shape for inference.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./out_dir",
    )
    parser.add_argument(
        "--with_p6",
        action="store_true",
        help="Whether your model uses p6 in FPN/PAN.",
    )
    return parser


def setup_kinect(args):
    # set up azure kinect
    if args.list:
        o3d.io.AzureKinectSensor.list_devices()
        exit()

    if args.config is not None:
        config = o3d.io.read_azure_kinect_sensor_config(args.config)
    else:
        config = o3d.io.AzureKinectSensorConfig()

    device = args.device
    if device < 0 or device > 255:
        print('Unsupported device id, fall back to 0')
        device = 0

    azure_kinect = o3d.io.AzureKinectSensor(config)
    if not azure_kinect.connect(device):
        raise RuntimeError('Failed to connect to sensor')

    return azure_kinect


def box_rgbd(origin_color, origin_depth, bounding_box: list):
    # get RGB-D image in bounding box
    bounding_box = [b if b > 0 else 0 for b in bounding_box]
    x0, y0, x1, y1 = map(int, bounding_box)
    np_color = np.asarray(origin_color)[y0: y1, x0: x1]
    np_depth = np.asarray(origin_depth)[y0: y1, x0: x1]
    new_color = o3d.geometry.Image(np_color.astype(np.uint8))
    new_depth = o3d.geometry.Image(np_depth.astype(np.uint16))
    return new_color, new_depth


def how_far_yolox(image, depth, args):
    # detection inference with yolox
    origin_img = copy.deepcopy(image)
    input_shape = tuple(map(int, args.input_shape.split(',')))
    img, ratio = preprocess(origin_img, input_shape)

    session = onnxruntime.InferenceSession(args.model)

    ort_inputs = {session.get_inputs()[0].name: img[None, :, :, :]}
    output = session.run(None, ort_inputs)
    predictions = demo_postprocess(output[0], input_shape, p6=args.with_p6)[0]

    boxes = predictions[:, :4]
    scores = predictions[:, 4:5] * predictions[:, 5:]

    boxes_xyxy = np.ones_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.
    boxes_xyxy /= ratio
    dets = multiclass_nms(boxes_xyxy, scores, nms_thr=0.45, score_thr=0.1)
    if dets is not None:
        # Final_boxes are list of bounding box (xyxy)
        final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
        origin_img = distance_vis(origin_img, depth, final_boxes, final_scores, final_cls_inds,
                         conf=args.score_thr, class_names=COCO_CLASSES)
    return origin_img


def distance_vis(img, depth, boxes, scores, cls_ids, conf=0.5, class_names=None):

    for i in range(len(boxes)):
        box = boxes[i]
        cls_id = int(cls_ids[i])
        score = scores[i]
        if score < conf:
            continue
        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])

        color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()
        box_color, box_depth = box_rgbd(img, depth, box)
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(box_color, box_depth,
                                                                        convert_rgb_to_intensity=False)
        color_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image,
            o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
        )
        point_clouds = np.asarray(color_pcd.points)
        if point_clouds.shape[0] == 0:
            continue
        distance = np.mean(np.sqrt(np.square(point_clouds[:, 0]) + np.square(point_clouds[:, 1]) + np.square(point_clouds[:, 2])))
        text = '{}:{:.1f}%+{:.3f}m'.format(class_names[cls_id], score * 100, distance)
        txt_color = (0, 0, 0) if np.mean(_COLORS[cls_id]) > 0.5 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX

        txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

        txt_bk_color = (_COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
        cv2.rectangle(
            img,
            (x0, y0 + 1),
            (x0 + txt_size[0] + 1, y0 + int(1.5*txt_size[1])),
            txt_bk_color,
            -1
        )
        cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)

    return img


_COLORS = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        0.000, 0.000, 0.000,
        0.143, 0.143, 0.143,
        0.286, 0.286, 0.286,
        0.429, 0.429, 0.429,
        0.571, 0.571, 0.571,
        0.714, 0.714, 0.714,
        0.857, 0.857, 0.857,
        0.000, 0.447, 0.741,
        0.314, 0.717, 0.741,
        0.50, 0.5, 0
    ]
).astype(np.float32).reshape(-1, 3)


if __name__ == '__main__':
    args = get_parser().parse_args()

    print('\n--------args----------')
    for k in list(vars(args).keys()):
        print('%s: %s' % (k, vars(args)[k]))
    print('--------args----------\n')

    # set up azure kinect
    azure_kinect = setup_kinect(args)


    def escape_callback():
        return False

    flag_exit = False
    glfw_key_escape = 256
    visualizer = o3d.visualization.VisualizerWithKeyCallback()
    visualizer.register_key_callback(glfw_key_escape, escape_callback)
    visualizer.create_window('RGBD', 1920, 540)

    vis_geometry_added = False
    while not flag_exit:
        rgbd = azure_kinect.capture_frame(True)

        if rgbd is None:
            continue

        det_image = how_far_yolox(np.asarray(rgbd.color), np.asarray(rgbd.depth), args)

        rgbd.color = o3d.geometry.Image(det_image.astype(np.uint8))

        if not vis_geometry_added:
            visualizer.add_geometry(rgbd)
            vis_geometry_added = True

        visualizer.update_geometry(rgbd)
        visualizer.poll_events()
        visualizer.update_renderer()
