# configure .yaml file to guide the model for training
yaml_text = """train: data/train/images
val: data/train/images

nc: 3
names: ['with_mask', 'mask_weared_incorrect', 'without_mask']"""

# 경로 맞춰줄것
with open("./working/yolov5/data/data.yaml", 'w') as file:
    file.write(yaml_text)

#%cat data/data.yaml