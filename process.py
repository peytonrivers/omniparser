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

yolo_model = get_yolo_model(model_path='weights/icon_detect/model.pt')
caption_model_processor = get_caption_model_processor(model_name="florence2", model_name_or_path="weights/icon_caption_florence")
# caption_model_processor = get_caption_model_processor(model_name="blip2", model_name_or_path="weights/icon_caption_blip2")

ocr = PaddleOCR(use_angle_cls=True, lang='en')

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
def process(
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
    return image, str(parsed_content_list)

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
    print(f"Image input class: {type(image_input)}")
    print(f"Image input: {image_input}")
    print(f"Box Threshold: {box_threshold}")
    print(f"Box threshold class: {type(box_threshold)}")
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
    current_time = time.perf_counter()
    print(f"Step 1 completed at: {current_time - start_time}")
    text, ocr_bbox = ocr_bbox_rslt
    current_time2 = time.perf_counter()
    print(f"Step 2 completed at: {current_time2 - start_time}")
    dino_labled_img, label_coordinates, parsed_content_list = get_som_labeled_img(image_input, yolo_model, BOX_TRESHOLD = box_threshold, output_coord_in_ratio=True, ocr_bbox=ocr_bbox,draw_bbox_config=draw_bbox_config, caption_model_processor=caption_model_processor, ocr_text=text,iou_threshold=iou_threshold, imgsz=imgsz,)
    boxes_details = []
    for i in range(len(parsed_content_list)):
        current_item = parsed_content_list[i]
        icon_item = {"icon": i}
        icon_item.update(current_item)
        boxes_details.append(icon_item)
        

    current_time3 = time.perf_counter()
    print(f"step 3 completed at: {current_time3 - start_time}")
    image = Image.open(io.BytesIO(base64.b64decode(dino_labled_img)))
    print('finish processing')
    image_width = image.width
    image_height = image.height

    for l in range(len(boxes_details)):
        current_box = boxes_details[l]
        current_coordinates = current_box["bbox"]
        content = current_box["content"]
        if not content:
            x1 = current_coordinates[0] * image_width
            y1 = current_coordinates[1] * image_height
            x2 = current_coordinates[2] * image_width
            y2 = current_coordinates[3] * image_height
            cropped_image = image.crop([x1, y1, x2, y2])
            cropped_image_array = np.array(cropped_image)
            result = ocr.ocr(cropped_image_array, cls=True)

            detected_text = ""

            if result and result[0]:
                words = []
                for line in result[0]:
                    words.append(line[1][0])   # line[1][0] is the recognized text
                detected_text = " ".join(words)
                print(detected_text)

            current_box["content"] = detected_text

    # parsed_content_list = str(parsed_content_list)
    print(f"Final Image: {image}")
    image.show()
    new_buffer = io.BytesIO()
    image.save(new_buffer, format="PNG")
    new_bytes = new_buffer.getvalue()
    new_sendable_bytes = base64.b64encode(new_bytes).decode("utf-8")
    return {"image": new_sendable_bytes, "bounding_boxes": boxes_details}

"""def image_processer(image_input,
    box_threshold,
    iou_threshold,
    use_paddleocr,
    imgsz
) -> Optional[Image.Image]:

    start_time = time.perf_counter()
    regular_bytes = base64.b64decode(image_input.encode("utf-8"))
    buffer = io.BytesIO(regular_bytes)
    image_input = Image.open(buffer)
    print(f"Image input class: {type(image_input)}")
    print(f"Image input: {image_input}")
    print(f"Box Threshold: {box_threshold}")
    print(f"Box threshold class: {type(box_threshold)}")
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
    current_time = time.perf_counter()
    print(f"Step 1 completed at: {current_time - start_time}")
    text, ocr_bbox = ocr_bbox_rslt
    current_time2 = time.perf_counter()
    print(f"Step 2 completed at: {current_time2 - start_time}")
    filtered_boxes, starting_idx = get_som_labeled_img(image_input, yolo_model, BOX_TRESHOLD = box_threshold, output_coord_in_ratio=True, ocr_bbox=ocr_bbox,draw_bbox_config=draw_bbox_config, caption_model_processor=caption_model_processor, ocr_text=text,iou_threshold=iou_threshold, imgsz=imgsz,)  
    print(f"filtered boxes: {filtered_boxes}")
    print(f"Starting Index: {starting_idx}")
    return {"filtered_boxes": filtered_boxes, "starting_idx": starting_idx}"""




"""
image_height = image.height
image_width = image.width

for i in range(len(bounding_boxes)):
    current_box = bounding_boxes[i]
    if not current_box["content"]:
        coordinates = current_box["bbox"]
        x1 = int(image_width * coordinates[0])
        y1 = int(image_height * coordinates[1])
        x2 = int(image_width * coordinates[2])
        y2 = int(image_height * coordinates[3])
        new_coordinates = [x1, y1, x2, y2]
        cropped_image = image.crop(new_coordinates)
        cropped_image_array = np.array(cropped_image)
        result = ocr.ocr(cropped_image_array, cls=True)
        text_list = []
        if result and result[0]:
            text_list = [line[1][0] for line in result[0]]

        # 2. Join all elements into one single line separated by a space
        single_line_text = " ".join(text_list)
        current_box["content"] = single_line_text
        
"""