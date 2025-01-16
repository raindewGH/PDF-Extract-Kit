# -*- coding: utf-8 -*-
# 2025-01-15 20:32:21
# raind

import os
import sys
import os.path as osp

sys.path.append(osp.join(os.path.dirname(
    os.path.abspath(__file__)), '..', '..', '..'))
from pdf_extract_kit.utils.config_loader import (load_config,
                                                 initialize_tasks_and_models)
from pdf_extract_kit.registry.registry import TASK_REGISTRY

TASK_NAME = 'img2layout'


def main(img, config_path):
    config = load_config(config_path)
    task_instances = initialize_tasks_and_models(config)

    layout_model = task_instances['layout_detection'].model if 'layout_detection' in task_instances else None

    pdf_extract_task = TASK_REGISTRY.get(TASK_NAME)(layout_model)
    extract_results = pdf_extract_task.process(img)

    print(extract_results)


def demo_img():
    from PIL import Image
    import numpy as np

    # 创建一个NumPy数组
    array = np.array([[255, 255, 255]], dtype=np.uint8)  # 这里是一个1x1的白色像素

    return Image.fromarray(array)


if __name__ == '__main__':
    config_path = '../configs/img2layout.yaml'
    img = demo_img()

    main(img, config_path=config_path)
