import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from PIL import Image
from pathlib import Path
from matplotlib.patches import Rectangle
from scipy.io import loadmat

devkit_path = Path('/home/user/workspace/image_classification_dataset/stanford_cars/devkit')
train_path = Path('/home/user/workspace/image_classification_dataset/stanford_cars/cars_train')
test_path = Path('/home/user/workspace/image_classification_dataset/stanford_cars/cars_test')

print(os.listdir(devkit_path))
cars_meta = loadmat(os.path.join(devkit_path, 'cars_meta.mat'))

labels_list = [str(c[0]) for c in cars_meta['class_names'][0]]
print(labels_list)