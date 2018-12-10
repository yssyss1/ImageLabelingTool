from PyQt5.QtGui import QImage
from lxml import etree, objectify

import cv2
from tqdm import tqdm
import os
import numpy as np
import xml.etree.ElementTree as ET


class ImageContainer:
    def __init__(self, image, fileName):
        self.image = image
        self.__fileName = fileName
        self.__imageWidth = image.width()
        self.__imageHeight = image.height()

    @property
    def fileName(self):
        return self.__fileName

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
    x_min, y_min, width, height = annotation["bbox"]
    return E.object(
            E.name(annotation["category_id"]),
            E.bndbox(
                E.xmin(x_min),
                E.ymin(y_min),
                E.xmax(x_min+width),
                E.ymax(y_min+height),
                ),
            )


#################################
#                               #
#   BELOW LINE IS TEST CODE     #
#                               #
#################################

def load_image(image_path):
    image = cv2.imread(image_path)
    image = np.array(image[:, :, ::-1])

    return image


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
    dataset_check('./test/images', './test/annotations', ['Ship', 'Buoy', 'Other'], 'test')