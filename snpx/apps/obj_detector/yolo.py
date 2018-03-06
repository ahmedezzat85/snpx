import os
import numpy as np
import cv2

#******************************
# YOLO Output Post Processing
#
#******************************

PRESETS = {
    'coco': { 'classes': [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
        "truck", "boat", "traffic light", "fire hydrant", "stop sign",
        "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
        "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
        "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
        "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
        "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
        "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
        "couch", "potted plant", "bed", "dining table", "toilet", "tv",
        "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
        "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
        "scissors", "teddy bear", "hair drier", "toothbrush"
    ], 'anchors': [[0.738768, 2.42204, 4.30971, 10.246, 12.6868],
                   [0.874946, 2.65704, 7.04493, 4.59428, 11.8741]]
    },
    'voc': { 'classes': [
        "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
        "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
        "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"],
        'anchors':  [[1.08, 3.42, 6.63, 9.42, 16.62],
                    [1.19, 4.41, 11.38, 5.11, 10.52]],
        "colors": [ 
                    [254.0, 254.0, 254], 
                    [239.88888888888889, 211.66666666666669, 127],
                    [225.77777777777777, 169.33333333333334, 0  ],
                    [211.66666666666669, 127.0, 254],
                    [197.55555555555557, 84.66666666666667, 127],
                    [183.44444444444443, 42.33333333333332, 0],
                    [169.33333333333334, 0.0, 254],
                    [155.22222222222223, -42.33333333333335, 127],
                    [141.11111111111111, -84.66666666666664, 0],
                    [127.0, 254.0, 254],
                    [112.88888888888889, 211.66666666666669, 127],
                    [98.77777777777777, 169.33333333333334, 0],
                    [84.66666666666667, 127.0, 254],
                    [70.55555555555556, 84.66666666666667, 127],
                    [56.44444444444444, 42.33333333333332, 0],
                    [42.33333333333332, 0.0, 254],
                    [28.222222222222236, -42.33333333333335, 127],
                    [14.111111111111118, -84.66666666666664, 0],
                    [0.0, 254.0, 254],
                    [-14.111111111111118, 211.66666666666669, 127]
                    ]
    }
}

def get_boxes(output, img_size, grid_size, num_boxes):
    """ extract bounding boxes from the last layer """

    w_img, h_img = img_size[1], img_size[0]
    boxes = np.reshape(output, (grid_size, grid_size, num_boxes, 4))
    offset = np.tile(np.arange(grid_size)[:, np.newaxis], (grid_size, 1, num_boxes))

    boxes[:, :, :, 0] += offset
    boxes[:, :, :, 1] += np.transpose(offset, (1, 0, 2))
    boxes[:, :, :, 0:2] /= 7.0
    # the predicted size is the square root of the box size
    boxes[:, :, :, 2:4] *= boxes[:, :, :, 2:4]
    boxes[:, :, :, [0, 2]] *= w_img
    boxes[:, :, :, [1, 3]] *= h_img
    return boxes

def parse_yolo_output_v1(output, img_size, num_classes):
    """ convert the output of the last fully connected layer (Darknet v1) """

    n_coord_box = 4    # number of coordinates in each bounding box
    grid_size = 7

    sc_offset = grid_size * grid_size * num_classes

    # autodetect num_boxes
    num_boxes = int((output.shape[0] - sc_offset) / (grid_size*grid_size*(n_coord_box+1)))
    box_offset = sc_offset + grid_size * grid_size * num_boxes
    class_probs = np.reshape(output[0:sc_offset], (grid_size, grid_size, num_classes))
    confidences = np.reshape(output[sc_offset:box_offset], (grid_size, grid_size, num_boxes))

    probs = np.zeros((grid_size, grid_size, num_boxes, num_classes))
    for i in range(num_boxes):
        for j in range(num_classes):
            probs[:, :, i, j] = class_probs[:, :, j] * confidences[:, :, i]

    boxes = get_boxes(output[box_offset:], img_size, grid_size, num_boxes)
    return boxes, probs

def logistic(val):
    """ compute the logistic activation """
    return 1.0 / (1.0 + np.exp(-val))


def softmax(val, axis=-1):
    """ compute the softmax of the given tensor, normalizing on axis """
    exp = np.exp(val - np.amax(val, axis=axis, keepdims=True))
    return exp / np.sum(exp, axis=axis, keepdims=True)

def get_boxes_v2(output, img_size, anchors):
    """ extract bounding boxes from the last layer (Darknet v2) """
    bias_w, bias_h = anchors

    w_img, h_img = img_size[1], img_size[0]
    grid_w, grid_h, num_boxes = output.shape[:3]

    # tweak: add a 0.5 offset to improve localization accuracy
    offset_x = np.tile(np.arange(grid_w)[:, np.newaxis], (grid_h, 1, num_boxes)) - 0.5
    offset_y = np.transpose(offset_x, (1, 0, 2))

    boxes = output.copy()
    boxes[:, :, :, 0] = (offset_x + logistic(boxes[:, :, :, 0])) / grid_w
    boxes[:, :, :, 1] = (offset_y + logistic(boxes[:, :, :, 1])) / grid_h
    boxes[:, :, :, 2] = np.exp(boxes[:, :, :, 2]) * bias_w / grid_w
    boxes[:, :, :, 3] = np.exp(boxes[:, :, :, 3]) * bias_h / grid_h

    boxes[:, :, :, [0, 2]] *= w_img
    boxes[:, :, :, [1, 3]] *= h_img

    return boxes


def parse_yolo_output_v2(output, img_size, num_classes, anchors):
    """ convert the output of the last convolutional layer (Darknet v2) """
    n_coord_box = 4

    # for each box: coordinates, probs scale, class probs
    num_boxes = output.shape[0] // (n_coord_box + 1 + num_classes)
    output = output.reshape((num_boxes, -1, output.shape[1], output.shape[2])).transpose((2, 3, 0, 1))
    probs = logistic(output[:, :, :, 4:5]) * softmax(output[:, :, :, 5:], axis=3)
    boxes = get_boxes_v2(output[:, :, :, :4], img_size, anchors)
    return boxes, probs


def parse_yolo_output(output, img_size, num_classes, anchors=None):
    """ convert the output of YOLO's last layer to boxes and confidence in each
    class """
    if len(output.shape) == 1:
        return parse_yolo_output_v1(output, img_size, num_classes)
    elif len(output.shape) == 3 and anchors is not None:
        return parse_yolo_output_v2(output, img_size, num_classes, anchors)
    else:
        raise ValueError(" output format not recognized")


def non_maxima_suppression(boxes, probs, classes_num, thr=0.2):
    """ greedily suppress low-scoring overlapped boxes """
    for i, box in enumerate(boxes):
        if probs[i] == 0:
            continue
        for j in range(i+1, len(boxes)):
            if classes_num[i] == classes_num[j] and iou(box, boxes[j]) > thr:
                probs[j] = 0.0

    return probs


def iou(box1, box2, denom="min"):
    """ compute intersection over union score """
    int_tb = min(box1[0]+0.5*box1[2], box2[0]+0.5*box2[2]) - \
             max(box1[0]-0.5*box1[2], box2[0]-0.5*box2[2])
    int_lr = min(box1[1]+0.5*box1[3], box2[1]+0.5*box2[3]) - \
             max(box1[1]-0.5*box1[3], box2[1]-0.5*box2[3])

    intersection = max(0.0, int_tb) * max(0.0, int_lr)
    area1, area2 = box1[2]*box1[3], box2[2]*box2[3]
    control_area = min(area1, area2) if denom == "min"  \
                   else area1 + area2 - intersection

    return intersection / control_area

def crop_max(img, shape):
    """ crop the largest dimension to avoid stretching """
    net_h, net_w = shape
    height, width = img.shape[:2]
    aratio = net_w / net_h

    if width > height * aratio:
        diff = int((width - height * aratio) / 2)
        return img[:, diff:-diff, :]
    else:
        diff = int((height - width / aratio) / 2)
        return img[diff:-diff, :, :]

def yolo_get_candidate_objects(output, img_size, mode='voc'):
    """ convert network output to bounding box predictions """

    threshold = 0.3
    iou_threshold = 0.4

    classes = PRESETS[mode]['classes']
    anchors = PRESETS[mode]['anchors']

    boxes, probs = parse_yolo_output(output, img_size, len(classes), anchors)

    filter_mat_probs = (probs >= threshold)
    filter_mat_boxes = np.nonzero(filter_mat_probs)[0:3]
    boxes_filtered = boxes[filter_mat_boxes]
    probs_filtered = probs[filter_mat_probs]
    classes_num_filtered = np.argmax(probs, axis=3)[filter_mat_boxes]

    idx = np.argsort(probs_filtered)[::-1]
    boxes_filtered = boxes_filtered[idx]
    probs_filtered = probs_filtered[idx]
    classes_num_filtered = classes_num_filtered[idx]

    # too many detections - exit
    if len(boxes_filtered) > 1e3:
        print("Too many detections, maybe an error? : {}".format(
            len(boxes_filtered)))
        return []

    probs_filtered = non_maxima_suppression(boxes_filtered, probs_filtered,
                                            classes_num_filtered, iou_threshold)
    filter_iou = (probs_filtered > 0.0)
    boxes_filtered = boxes_filtered[filter_iou]
    probs_filtered = probs_filtered[filter_iou]
    classes_num_filtered = classes_num_filtered[filter_iou]

    result = []
    h, w, _ = img_size
    for class_id, box, prob in zip(classes_num_filtered, boxes_filtered, probs_filtered):
        x_start, x_end = max(int(box[0]) - int(box[2])//2, 0), min(int(box[0]) + int(box[2])//2, w)
        y_start, y_end = max(int(box[1]) - int(box[3])//2, 0), min(int(box[1]) + int(box[3])//2, h)
        result.append([classes[class_id], x_start, x_end, y_start, y_end, prob])

    return result