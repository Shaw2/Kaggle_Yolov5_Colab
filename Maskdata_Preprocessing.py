# Import required libraries
import pandas as pd
import numpy as np
import os
import glob
# from datetime import datetime
import xml.etree.ElementTree as ET
# import cv2
# import matplotlib.pyplot as plt
# import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

input_data = './face-mask-detection'
output_data = './working'

annotations_path = './face-mask-detection/annotations'
images_path = './face-mask-detection/images'

dataset = {
    "file": [],
    "name": [],
    "width": [],
    "height": [],
    "xmin": [],
    "ymin": [],
    "xmax": [],
    "ymax": [],
}

for anno in glob.glob(annotations_path + "/*.xml"):
    tree = ET.parse(anno)

    for elem in tree.iter():
        if 'size' in elem.tag:
            for attr in list(elem):
                if 'width' in attr.tag:
                    width = int(round(float(attr.text)))
                if 'height' in attr.tag:
                    height = int(round(float(attr.text)))

        if 'object' in elem.tag:
            for attr in list(elem):

                if 'name' in attr.tag:
                    name = attr.text
                    dataset['name'] += [name]
                    dataset['width'] += [width]
                    dataset['height'] += [height]
                    dataset['file'] += [anno.split('/')[-1][0:-4]]

                if 'bndbox' in attr.tag:
                    for dim in list(attr):
                        if 'xmin' in dim.tag:
                            xmin = int(round(float(dim.text)))
                            dataset['xmin'] += [xmin]
                        if 'ymin' in dim.tag:
                            ymin = int(round(float(dim.text)))
                            dataset['ymin'] += [ymin]
                        if 'xmax' in dim.tag:
                            xmax = int(round(float(dim.text)))
                            dataset['xmax'] += [xmax]
                        if 'ymax' in dim.tag:
                            ymax = int(round(float(dim.text)))
                            dataset['ymax'] += [ymax]

df=pd.DataFrame(dataset)
print(df.head())

name_dict = {
    'with_mask': 0,
    'mask_weared_incorrect': 1,
    'without_mask': 2
}

df['class'] = df['name'].map(name_dict)

print(np.sort(df.name.unique()))

# os 앞에 *을 넣어주는 것과 아닌 것의 차이가 존재
fileNames = [*os.listdir('./face-mask-detection/images')]
print('There are {} images in the dataset'.format(len(fileNames)))

from sklearn.model_selection import train_test_split
train, test = train_test_split(fileNames, test_size=0.1, random_state=22)
test, val = train_test_split(test, test_size=0.7, random_state=22)

print("Length of Train =",len(train))
print("="*30)
print("Length of Valid =",len(val))
print("="*30)
print("Length of test =", len(test))

os.chdir('./working/')
print(os.getcwd())
os.makedirs('./yolov5/data/train', exist_ok=True)
os.makedirs('./yolov5/data/val', exist_ok=True)
os.makedirs('./yolov5/data/test', exist_ok=True)
os.makedirs('./yolov5/data/train/images', exist_ok=True)
os.makedirs('./yolov5/data/train/labels', exist_ok=True)
os.makedirs('./yolov5/data/test/images', exist_ok=True)
os.makedirs('./yolov5/data/test/labels', exist_ok=True)
os.makedirs('./yolov5/data/val/images', exist_ok=True)
os.makedirs('./yolov5/data/val/labels', exist_ok=True)

from PIL import Image

os.chdir('..')

def copyImages(imageList, folder_Name):
    for image in imageList:
        img = Image.open(input_data+"/images/"+image)
        img1 = img.resize((640, 480))
        _ = img1.save(output_data+"/yolov5/data/"+folder_Name+"/images/"+image)

# copyImages(train, "train")
# copyImages(val, "val")
# copyImages(test, "test")

print(df.head())

df['xmax'] = (640/df['width'])*df['xmax']
df['ymax'] = (480/df['height'])*df['ymax']
df['xmin'] = (640/df['width'])*df['xmin']
df['ymin'] = (480/df['height'])*df['ymin']

df[['xmax', 'ymax', 'xmin', 'ymin']] = df[['xmax', 'ymax', 'xmin', 'ymin']].astype('int64')

df['x_center'] = (df['xmax']+df['xmin'])/(2*640)
df['y_center'] = (df['ymax']+df['ymin'])/(2*480)
df['box_height'] = (df['xmax']-df['xmin'])/(640)
df['box_width'] = (df['ymax']-df['ymin'])/(480)

print(df.head())

df = df.astype('string')

print('=='*30)
print(df)

print(train)

def create_labels(image_list, data_name):
    fileNames = [x.split(".")[0] for x in image_list]
    print('fileNames : ', fileNames)
    for name in fileNames:
        data = df[df.file == name]
        print('data : ', data, 'data-END')
        box_list = []

        for index in range(len(data)):
            row = data.iloc[index]
            box_list.append(row['class'] + " " + row["x_center"] + " " + row["y_center"] \
                            + " " + row["box_height"] + " " + row["box_width"])

        text = "\n".join(box_list)
        with open(output_data + "/yolov5/data/" + data_name + "/labels/" + name + ".txt", "w") as file:
            file.write(text)


create_labels(train, "train")
create_labels(val, "val")
create_labels(test, "test")