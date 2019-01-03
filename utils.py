from lxml import objectify

import cv2
from tqdm import tqdm
import numpy as np
import xml.etree.ElementTree as ET
import os
from glob import glob


class ImageContainer:
    def __init__(self, image, filePath):
        self.__image = image
        self.__imageWidth = image.width() if image is not None else 0
        self.__imageHeight = image.height() if image is not None else 0
        self.__filePath = filePath

    @property
    def image(self):
        return self.__image

    @property
    def filePath(self):
        return self.__filePath

    @property
    def fileName(self):
        # Add '\\' as splitter for window directory path
        return self.__filePath.split('\\')[-1].split('/')[-1]

    @property
    def imageWidth(self):
        return self.__imageWidth

    @property
    def imageHeight(self):
        return self.__imageHeight


def xml_root(filename, height, width):
    E = objectify.ElementMaker(annotate=False)
    return E.annotation(
        E.filename(filename),
        E.size(
            E.width(width),
            E.height(height),
            E.depth(3),
        )
    )


def instance_to_xml(annotation):
    E = objectify.ElementMaker(annotate=False)
    x_min, y_min, x_max, y_max = annotation["bbox"]
    return E.object(
            E.name(annotation["category_id"]),
            E.bndbox(
                E.xmin(x_min),
                E.ymin(y_min),
                E.xmax(x_max),
                E.ymax(y_max),
                ),
            )


def globWithTypes(path, exts):
    path = os.path.join(path, "*")
    filePath = []
    for files in [glob(path+ext) for ext in exts]:
        for file in files:
            filePath.append(file)
    return filePath


################################################
#                                              #
#   BELOW LINE IS Yolo Postprocessing CODE     #
#                                              #
################################################


class BoundBox:
    def __init__(self, xmin, ymin, xmax, ymax, confidence=None, classes=None):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax

        self.confidence = confidence
        self.classes = classes

        self.label = -1
        self.score = -1

    def get_label(self):
        if self.label == -1:
            self.label = np.argmax(self.classes)

        return self.label

    def get_score(self):
        if self.score == -1:
            self.score = self.classes[self.get_label()]

        return self.score


def prediction(image,
               model,
               obj_threshold=0.3,
               nms_threshold=0.3,
               image_width=416,
               image_height=416,
               grid_h=13,
               grid_w=13,
               box_num=5,
               normalize=True,
               anchors=None
               ):

    if anchors is None:
        anchors = [0.78353, 1.57529, 1.02559, 0.65428, 1.97076, 1.00357, 3.76925, 2.32570, 0.35109, 0.39320]

    input_image = cv2.resize(image, (image_height, image_width))
    input_image = input_image / 255. if normalize else input_image
    input_image = np.expand_dims(input_image, 0)

    netout = model.predict(input_image)

    boxes = decode_netout(netout[0],
                          shape_dims=(grid_h, grid_w, box_num, 4 + 1 + 1),
                          anchors=anchors,
                          nb_class=1,
                          obj_threshold=obj_threshold,
                          nms_threshold=nms_threshold
                          )

    bouding_boxes = get_bounding_boxes(image, boxes, grid_h, grid_w)

    return bouding_boxes


def load_image(image_path):
    image = cv2.imread(image_path)
    image = np.array(image[..., ::-1])

    return image


def decode_netout(netout, shape_dims, anchors, nb_class, obj_threshold=0.3, nms_threshold=0.3):
    netout = np.reshape(netout, shape_dims)
    grid_h, grid_w, nb_box = netout.shape[:3]

    boxes = []

    netout[..., 4] = sigmoid(netout[..., 4])
    netout[..., 5] = netout[..., 4] * sigmoid(netout[..., 5])
    netout[..., 5] *= netout[..., 5] > obj_threshold

    for row in range(grid_h):
        for col in range(grid_w):
            for b in range(nb_box):

                classes = netout[row, col, b, 5:]

                if np.sum(classes) > 0:
                    x, y, w, h, confidence = netout[row, col, b, :5]

                    x = (col + sigmoid(x))
                    y = (row + sigmoid(y))
                    w = anchors[2 * b + 0] * np.exp(w)
                    h = anchors[2 * b + 1] * np.exp(h)

                    box = BoundBox(x - w / 2, y - h / 2, x + w / 2, y + h / 2, confidence, classes)

                    boxes.append(box)

    for c in range(nb_class):
        sorted_indices = list(reversed(np.argsort([box.classes[c] for box in boxes])))

        for i in range(len(sorted_indices)):
            index_i = sorted_indices[i]

            if boxes[index_i].classes[c] == 0:
                continue
            else:
                for j in range(i + 1, len(sorted_indices)):
                    index_j = sorted_indices[j]

                    if bbox_iou(boxes[index_i], boxes[index_j]) >= nms_threshold:
                        boxes[index_j].classes[c] = 0

    boxes = [box for box in boxes if box.get_score() > 0]

    return boxes


def get_bounding_boxes(image, boxes, grid_h, grid_w):
    image_h, image_w, _ = image.shape
    box_list = []

    for box in boxes:
        xmin = max(int(box.xmin * image_w / grid_w), 0)
        ymin = max(int(box.ymin * image_h / grid_h), 0)
        xmax = min(int(box.xmax * image_w / grid_w), image_w)
        ymax = min(int(box.ymax * image_h / grid_h), image_h)

        box_list.append([xmin, ymin, xmax-xmin, ymax-ymin])

    return box_list


def sigmoid(x):
    x = np.array(x)
    return 1. / (1. + np.exp(-x))


def bbox_iou(box1: BoundBox, box2: BoundBox):
    intersect_w = interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
    intersect_h = interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])

    intersect = intersect_w * intersect_h

    w1, h1 = box1.xmax - box1.xmin, box1.ymax - box1.ymin
    w2, h2 = box2.xmax - box2.xmin, box2.ymax - box2.ymin

    union = w1 * h1 + w2 * h2 - intersect

    return float(intersect) / union


def interval_overlap(interval_a, interval_b):
    x1, x2 = interval_a
    x3, x4 = interval_b

    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2, x4) - x1
    else:
        if x2 < x3:
            return 0
        else:
            return min(x2, x4) - x3


#################################
#                               #
#   BELOW LINE IS TEST CODE     #
#                               #
#################################


def dataset_check(image_dir, xml_dir, labels, name):

    if not os.path.exists(image_dir):
        raise FileNotFoundError('{} is not exists'.format(image_dir))

    if not os.path.exists(xml_dir):
        raise FileNotFoundError('{} is not exists'.format(image_dir))

    os.makedirs('./dataset_check', exist_ok=True)
    print('Start check {} dataset'.format(name))

    instances, _ = parse_annotation(xml_dir, image_dir, labels, name)
    idx = 0
    for instance in tqdm(instances, desc='Check {} dataset'.format(name)):
        idx += 1

        image_path = instance['filename']
        image = load_image(image_path)

        for object in instance['object']:
            cv2.rectangle(image, (object['xmin'], object['ymin']), (object['xmax'], object['ymax']), (0, 255, 0), 2)
            cv2.putText(image,
                        object['name'],
                        (object['xmin'], object['ymin'] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1e-3 * image.shape[0],
                        (0, 255, 0), 1)

        cv2.imwrite('./dataset_check/{}.jpg'.format(idx), image[:, :, ::-1])

    print('End check {} dataset!'.format(name))


def parse_annotation(ann_dir, img_dir, labels, data_name):
    if len(labels) == 0:
        raise ValueError("given label is not valid")

    print("Start Parsing {} data annotions...".format(data_name))

    all_imgs = []
    seen_labels = {}

    for ann in tqdm(sorted(os.listdir(ann_dir)), desc="Parse {} annotations".format(data_name)):
        img = {"object": []}

        tree = ET.parse(os.path.join(ann_dir, ann))

        for elem in tree.iter():
            if "filename" in elem.tag:
                img["filename"] = os.path.join(img_dir, elem.text)
            if "width" in elem.tag:
                img["width"] = int(elem.text)
            if "height" in elem.tag:
                img["height"] = int(elem.text)
            if "object" in elem.tag or "part" in elem.tag:
                obj = {}

                for attr in list(elem):
                    if "name" in attr.tag:
                        obj["name"] = attr.text

                        if obj["name"] in seen_labels:
                            seen_labels[obj["name"]] += 1
                        else:
                            seen_labels[obj["name"]] = 1

                        if len(labels) > 0 and obj["name"] not in labels:
                            break
                        else:
                            img["object"] += [obj]

                    if "bndbox" in attr.tag:
                        for dim in list(attr):
                            if "xmin" in dim.tag:
                                obj["xmin"] = int(round(float(dim.text)))
                            if "ymin" in dim.tag:
                                obj["ymin"] = int(round(float(dim.text)))
                            if "xmax" in dim.tag:
                                obj["xmax"] = int(round(float(dim.text)))
                            if "ymax" in dim.tag:
                                obj["ymax"] = int(round(float(dim.text)))

        if len(img["object"]) > 0:
            all_imgs += [img]

    print("End Parsing Annotations!")

    return all_imgs, seen_labels


if __name__ == '__main__':
    dataset_check('./MVI_0788_VIS_OB/image',
                  './MVI_0788_VIS_OB/annotation',
                  ['Ship', 'Speed boat', 'Sail boat', 'Buoy', 'Other'],
                  'test')
    # globWithTypes('./types', ['png', 'jpg', 'jpeg'])