import numpy as np
import sys
sys.path.insert(0, '.')
from lib.base_dataset import as_same_class

# クラスすうに柔軟性を持たせるにあたり、signate のgrey scale の値の若い順にした。
labels_info = [
    {"hasInstances": False, "category": "void", "catid": 0, "name": "car", "ignoreInEval": True, "id": 0, "color": [0, 0, 255], "trainId": 0},
    {"hasInstances": False, "category": "construction", "catid": 2, "name": "lane", "ignoreInEval": False, "id": 13, "color": [69, 47, 142], "trainId": 13},
    {"hasInstances": False, "category": "void", "catid": 0, "name": "truck", "ignoreInEval": True, "id": 2, "color": [180, 0, 129], "trainId": 2},
    {"hasInstances": False, "category": "object", "catid": 3, "name": "own", "ignoreInEval": False, "id": 19, "color": [86, 62, 67], "trainId": 19},
    {"hasInstances": False, "category": "construction", "catid": 2, "name": "ground", "ignoreInEval": True, "id": 14, "color": [136, 45, 66], "trainId": 14},
    {"hasInstances": False, "category": "void", "catid": 0, "name": "pedestrian", "ignoreInEval": True, "id": 4, "color": [255, 0, 0], "trainId": 4},
    {"hasInstances": False, "category": "object", "catid": 3, "name": "others", "ignoreInEval": True, "id": 18, "color": [82, 99, 0], "trainId": 18},
    {"hasInstances": False, "category": "construction", "catid": 2, "name": "roadshoulder", "ignoreInEval": True, "id": 16, "color": [215, 0, 255], "trainId": 16},
    {"hasInstances": False, "category": "flat", "catid": 1, "name": "sky", "ignoreInEval": True, "id": 9, "color": [0, 152, 225], "trainId": 9},
    {"hasInstances": False, "category": "void", "catid": 0, "name": "motorbike", "ignoreInEval": True, "id": 5, "color": [65, 166, 1], "trainId": 5},
    {"hasInstances": False, "category": "construction", "catid": 2, "name": "wall", "ignoreInEval": False, "id": 12, "color": [92, 136, 125], "trainId": 12},
    {"hasInstances": False, "category": "flat", "catid": 1, "name": "building", "ignoreInEval": True, "id": 10, "color": [0, 203, 151], "trainId": 10},
    {"hasInstances": False, "category": "object", "catid": 3, "name": "obstacle", "ignoreInEval": False, "id": 17, "color": [180, 131, 135], "trainId": 17},
    {"hasInstances": False, "category": "void", "catid": 0, "name": "bicycle", "ignoreInEval": True, "id": 6, "color": [208, 149, 1], "trainId": 6},
    {"hasInstances": False, "category": "flat", "catid": 1, "name": "signs", "ignoreInEval": False, "id": 8, "color": [255, 134, 0], "trainId": 8},
    {"hasInstances": False, "category": "void", "catid": 0, "name": "svehicle", "ignoreInEval": True, "id": 3, "color": [255, 121, 166], "trainId": 3},
    {"hasInstances": False, "category": "construction", "catid": 2, "name": "sidewalk", "ignoreInEval": True, "id": 15, "color": [0, 255, 255], "trainId": 15},
    {"hasInstances": False, "category": "construction", "catid": 2, "name": "natural", "ignoreInEval": False, "id": 11, "color": [85, 255, 50], "trainId": 11},
    {"hasInstances": False, "category": "void", "catid": 0, "name": "bus", "ignoreInEval": True, "id": 1, "color": [193, 214, 0], "trainId": 1},
    {"hasInstances": False, "category": "flat", "catid": 1, "name": "signal", "ignoreInEval": False, "id": 7, "color": [255, 255, 0], "trainId": 7},
]

def get_palette(num_class):
    remap = as_same_class[num_class][1:]
    palette = np.zeros((20, 3), dtype=int)
    for i in range(len(remap)):
        palette[remap[i]] = labels_info[i]["color"]
    return palette