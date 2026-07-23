from typing import Optional

import gradio as gr
import numpy as np
import torch
from PIL import Image
import io
import requests

import base64, os
from util.utils import check_ocr_box, get_yolo_model, get_caption_model_processor, get_som_labeled_img
import torch
from PIL import Image
import time
from paddleocr import PaddleOCR
import json

print("Loading YOLO...")
yolo_model = get_yolo_model(model_path='weights/icon_detect/model.pt')

"""print("Loading Florence...")
caption_model_processor = get_caption_model_processor(
    model_name="florence2",
    model_name_or_path="weights/icon_caption_florence"
)"""



MARKDOWN = """
# OmniParser for Pure Vision Based General GUI Agent 🔥
<div>
    <a href="https://arxiv.org/pdf/2408.00203">
        <img src="https://img.shields.io/badge/arXiv-2408.00203-b31b1b.svg" alt="Arxiv" style="display:inline-block;">
    </a>
</div>

OmniParser is a screen parsing tool to convert general GUI screen to structured elements. 
"""

DEVICE = torch.device('cuda')

# @spaces.GPU
# @torch.inference_mode()
# @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
"""def process(
    image_input,
    box_threshold,
    iou_threshold,
    use_paddleocr,
    imgsz
) -> Optional[Image.Image]:

    print(f"Image input: {image_input}")
    print(f"Box Threshold: {box_threshold}")
    print(f"Iou threshold: {iou_threshold}")
    print(f"Use Paddleocr: {use_paddleocr}")
    print(f"Images: {imgsz}")
    box_overlay_ratio = image_input.size[0] / 3200
    draw_bbox_config = {
        'text_scale': 0.8 * box_overlay_ratio,
        'text_thickness': max(int(2 * box_overlay_ratio), 1),
        'text_padding': max(int(3 * box_overlay_ratio), 1),
        'thickness': max(int(3 * box_overlay_ratio), 1),
    }
    # import pdb; pdb.set_trace()

    ocr_bbox_rslt, is_goal_filtered = check_ocr_box(image_input, display_img = False, output_bb_format='xyxy', goal_filtering=None, easyocr_args={'paragraph': False, 'text_threshold':0.9}, use_paddleocr=use_paddleocr)
    text, ocr_bbox = ocr_bbox_rslt
    dino_labled_img, label_coordinates, parsed_content_list = get_som_labeled_img(image_input, yolo_model, BOX_TRESHOLD = box_threshold, output_coord_in_ratio=True, ocr_bbox=ocr_bbox,draw_bbox_config=draw_bbox_config, caption_model_processor=caption_model_processor, ocr_text=text,iou_threshold=iou_threshold, imgsz=imgsz,)  
    image = Image.open(io.BytesIO(base64.b64decode(dino_labled_img)))
    print('finish processing')
    parsed_content_list = '\n'.join([f'icon {i}: ' + str(v) for i,v in enumerate(parsed_content_list)])
    # parsed_content_list = str(parsed_content_list)
    print(f"Final Image: {image}")
    print(f"Parsed Content: {str(parsed_content_list)}")
    return image, str(parsed_content_list)"""

def image_processer(image_input,
    box_threshold,
    iou_threshold,
    use_paddleocr,
    imgsz
) -> Optional[Image.Image]:

    start_time = time.perf_counter()
    regular_bytes = base64.b64decode(image_input.encode("utf-8"))
    buffer = io.BytesIO(regular_bytes)
    image_input = Image.open(buffer)
    print(f"Image input: {image_input}")
    print(f"Box Threshold: {box_threshold}")
    print(f"Iou threshold: {iou_threshold}")
    print(f"Use Paddleocr: {use_paddleocr}")
    print(f"Images: {imgsz}")
    box_overlay_ratio = image_input.size[0] / 3200
    draw_bbox_config = {
        'text_scale': 0.8 * box_overlay_ratio,
        'text_thickness': max(int(2 * box_overlay_ratio), 1),
        'text_padding': max(int(3 * box_overlay_ratio), 1),
        'thickness': max(int(3 * box_overlay_ratio), 1),
    }
    # import pdb; pdb.set_trace()

    ocr_bbox_rslt, is_goal_filtered = check_ocr_box(image_input, display_img = False, output_bb_format='xyxy', goal_filtering=None, easyocr_args={'paragraph': False, 'text_threshold':0.9}, use_paddleocr=use_paddleocr)
    print(f"OCR bbox result: {ocr_bbox_rslt}")
    print(f"Is goal filtered: {is_goal_filtered}")
    current_time = time.perf_counter()
    print(f"Step 1 completed at: {current_time - start_time}")
    text, ocr_bbox = ocr_bbox_rslt
    current_time2 = time.perf_counter()
    print(f"Step 2 completed at: {current_time2 - start_time}")
    dino_labled_img, label_coordinates, parsed_content_list = get_som_labeled_img(image_input, yolo_model, BOX_TRESHOLD = box_threshold, output_coord_in_ratio=True, ocr_bbox=ocr_bbox,draw_bbox_config=draw_bbox_config, ocr_text=text,iou_threshold=iou_threshold, imgsz=imgsz,)
    boxes_details = []
    for i in range(len(parsed_content_list)):
        current_item = parsed_content_list[i]
        icon_item = {"icon": i}
        icon_item.update(current_item)
        boxes_details.append(icon_item)
        
    print(f"Final boxes details: {boxes_details}")
    current_time3 = time.perf_counter()
    print(f"step 3 completed at: {current_time3 - start_time}")
    image = Image.open(io.BytesIO(base64.b64decode(dino_labled_img)))
    print('finish processing')

    # parsed_content_list = str(parsed_content_list)
    print(f"Final Image: {image}")
    image.show()
    new_buffer = io.BytesIO()
    image.save(new_buffer, format="PNG")
    new_bytes = new_buffer.getvalue()
    new_sendable_bytes = base64.b64encode(new_bytes).decode("utf-8")
    ending_time = time.perf_counter() - start_time
    return {"image": new_sendable_bytes, "bounding_boxes": boxes_details, "time": ending_time}
