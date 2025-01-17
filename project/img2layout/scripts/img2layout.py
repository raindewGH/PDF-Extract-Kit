# -*- coding: utf-8 -*-
# 2025-01-15 20:32:21
# raind

import os
import sys
import os.path as osp
import gc
import torch

sys.path.append(osp.join(os.path.dirname(
    os.path.abspath(__file__)), '..', '..', '..'))
from pdf_extract_kit.tasks.base_task import BaseTask
from pdf_extract_kit.registry.registry import TASK_REGISTRY


@TASK_REGISTRY.register('img2layout')
class Img2Layout(BaseTask):
    def __init__(self, layout_model):
        self.layout_model = layout_model

    def convert_format(self, yolo_res, id_to_names, ):
        """
        convert yolo format to pdf-extract format.
        """
        res_list = []
        for xyxy, conf, cla in zip(yolo_res.boxes.xyxy.cpu(), yolo_res.boxes.conf.cpu(), yolo_res.boxes.cls.cpu()):
            xmin, ymin, xmax, ymax = [int(p.item()) for p in xyxy]
            new_item = {
                'category_type': id_to_names[int(cla.item())],
                'poly': [xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax],
                'score': round(float(conf.item()), 2),
            }
            res_list.append(new_item)
        return res_list

    def process(self, img):
        # 预测
        if self.layout_model is not None:
            ori_layout_res = self.layout_model.predict([img], "")[0]
            layout_res = self.convert_format(ori_layout_res, self.layout_model.id_to_names)
        else:
            layout_res = []

        # 释放内存
        torch.cuda.empty_cache()
        gc.collect()

        return layout_res
