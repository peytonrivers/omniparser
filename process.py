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

file_path = "/Users/peytonrivers/Desktop/test_screenshot.png"

image = Image.open(file_path)

buffer = io.BytesIO()

image.save(buffer, format="PNG")

bytes = buffer.getvalue()

encoded_bytes = base64.b64encode(bytes).decode("utf-8")


image_processer(encoded_bytes, 0.05, 0.10, True, 640)

"""
Check ocr box full text and coordinate results: [[[[98.0, 15.0], [221.0, 15.0], [221.0, 57.0], [98.0, 57.0]], ('Chrome', 0.9579596519470215)], [[[242.0, 15.0], [300.0, 15.0], [300.0, 51.0], [242.0, 51.0]], ('File', 0.8519994020462036)], [[[331.0, 18.0], [389.0, 18.0], [389.0, 51.0], [331.0, 51.0]], ('Edit', 0.9827934503555298)], [[[416.0, 15.0], [487.0, 15.0], [487.0, 51.0], [416.0, 51.0]], ('View', 0.9371057748794556)], [[[511.0, 15.0], [619.0, 15.0], [619.0, 57.0], [511.0, 57.0]], ('History', 0.8998951315879822)], [[[643.0, 18.0], [793.0, 18.0], [793.0, 51.0], [643.0, 51.0]], ('Bookmarks', 0.9799990057945251)], [[[818.0, 18.0], [925.0, 18.0], [925.0, 51.0], [818.0, 51.0]], ('Profiles', 0.9863989353179932)], [[[952.0, 18.0], [1008.0, 18.0], [1008.0, 51.0], [952.0, 51.0]], ('Tab', 0.9799025058746338)], [[[1035.0, 18.0], [1148.0, 18.0], [1148.0, 51.0], [1035.0, 51.0]], ('Window', 0.957953155040741)], [[[1164.0, 18.0], [1243.0, 18.0], [1243.0, 54.0], [1164.0, 54.0]], ('Help', 0.9343386888504028)], [[[2652.0, 12.0], [2916.0, 12.0], [2916.0, 54.0], [2652.0, 54.0]], ('Wed Jul 15 3:46 PM', 0.9485848546028137)], [[[591.0, 93.0], [671.0, 93.0], [671.0, 125.0], [591.0, 125.0]], ('C) p1', 0.5751838684082031)], [[[710.0, 96.0], [744.0, 96.0], [744.0, 122.0], [710.0, 122.0]], ('G', 0.9470754265785217)], [[[812.0, 90.0], [879.0, 90.0], [879.0, 125.0], [812.0, 125.0]], ('G p', 0.8284731507301331)], [[[1029.0, 93.0], [1099.0, 93.0], [1099.0, 125.0], [1029.0, 125.0]], ('8H', 0.683895468711853)], [[[2720.0, 87.0], [2912.0, 87.0], [2912.0, 128.0], [2720.0, 128.0]], ('+ Ask Gemini', 0.943070113658905)], [[[306.0, 173.0], [916.0, 170.0], [916.0, 212.0], [306.0, 215.0]], ('allstate.jobs/job-not-found/?jobid=23310874', 0.9991081357002258)], [[[2664.0, 173.0], [2909.0, 170.0], [2910.0, 212.0], [2665.0, 215.0]], ('Action Required :', 0.9601717591285706)], [[[119.0, 251.0], [343.0, 251.0], [343.0, 284.0], [119.0, 284.0]], ('C @DogeTehDoge', 0.9169096946716309)], [[[368.0, 248.0], [631.0, 248.0], [631.0, 287.0], [368.0, 287.0]], ('G john d rockefeller...', 0.9854105114936829)], [[[680.0, 248.0], [943.0, 248.0], [943.0, 287.0], [680.0, 287.0]], (' 28" Outdoor Prop...', 0.9385143518447876)], [[[983.0, 248.0], [1262.0, 248.0], [1262.0, 287.0], [983.0, 287.0]], ('G back cracker - Go..', 0.9501729607582092)], [[[1298.0, 251.0], [1415.0, 251.0], [1415.0, 284.0], [1298.0, 284.0]], ('M Gmail', 0.8953631520271301)], [[[1433.0, 245.0], [1715.0, 245.0], [1715.0, 284.0], [1433.0, 284.0]], (' Image result for gy...', 0.9610952138900757)], [[[1743.0, 251.0], [1899.0, 251.0], [1899.0, 284.0], [1743.0, 284.0]], ('D YouTube', 0.9132731556892395)], [[[1908.0, 251.0], [2037.0, 251.0], [2037.0, 284.0], [1908.0, 284.0]], ('QMaps', 0.8731428384780884)], [[[2052.0, 248.0], [2269.0, 248.0], [2269.0, 281.0], [2052.0, 281.0]], (' Import to Loox', 0.977820098400116)], [[[2297.0, 248.0], [2511.0, 248.0], [2511.0, 281.0], [2297.0, 281.0]], (' Import to Loox', 0.9519518613815308)], [[[2851.0, 254.0], [2885.0, 254.0], [2885.0, 281.0], [2851.0, 281.0]], ('>>', 0.7894951701164246)], [[[2491.0, 331.0], [2653.0, 338.0], [2651.0, 377.0], [2489.0, 370.0]], ('Allstate.com', 0.8537144660949707)], [[[285.0, 397.0], [524.0, 397.0], [524.0, 448.0], [285.0, 448.0]], ('SAllstate', 0.9022436141967773)], [[[1332.0, 412.0], [1494.0, 412.0], [1494.0, 454.0], [1332.0, 454.0]], ('About v', 0.8857163786888123)], [[[1553.0, 415.0], [1752.0, 415.0], [1752.0, 448.0], [1553.0, 448.0]], ('Our Impact', 0.9777528047561646)], [[[1859.0, 415.0], [2064.0, 415.0], [2064.0, 448.0], [1859.0, 448.0]], ('Investors v', 0.9280672669410706)], [[[2132.0, 418.0], [2315.0, 418.0], [2315.0, 451.0], [2132.0, 451.0]], ('Careersv', 0.9736491441726685)], [[[2376.0, 415.0], [2689.0, 415.0], [2689.0, 454.0], [2376.0, 454.0]], ('News & Stories v', 0.9616782069206238)], [[[389.0, 702.0], [1054.0, 702.0], [1054.0, 786.0], [389.0, 786.0]], ('Page not found', 0.9856423139572144)], [[[377.0, 819.0], [2043.0, 822.0], [2043.0, 914.0], [377.0, 911.0]], ("Don't worry, you're still in Good Hands.", 0.9866601824760437)], [[[371.0, 947.0], [2358.0, 953.0], [2358.0, 1004.0], [371.0, 998.0]], ("While you're here, learn about job opportunities with our global brands, competitive employee benefits and all the other", 0.9821390509605408)], [[[374.0, 1001.0], [928.0, 998.0], [928.0, 1040.0], [374.0, 1043.0]], ('great reasons to work at Allstate.', 0.9533141851425171)], [[[426.0, 1081.0], [784.0, 1081.0], [784.0, 1123.0], [426.0, 1123.0]], ('Search jobs by keyword:', 0.969714343547821)], [[[977.0, 1081.0], [1259.0, 1081.0], [1259.0, 1123.0], [977.0, 1123.0]], ('Select career area:', 0.9875247478485107)], [[[444.0, 1153.0], [864.0, 1153.0], [864.0, 1192.0], [444.0, 1192.0]], ('Enter a Job or Keyword', 0.9786590933799744)], [[[996.0, 1147.0], [1262.0, 1150.0], [1261.0, 1192.0], [995.0, 1189.0]], ('All Departments', 0.9903029203414917)], [[[1635.0, 1145.0], [1815.0, 1138.0], [1817.0, 1180.0], [1637.0, 1187.0]], (' search jobs', 0.879134476184845)], [[[383.0, 1461.0], [1191.0, 1461.0], [1191.0, 1530.0], [383.0, 1530.0]], ('Search jobs by career area', 0.9977526068687439)], [[[380.0, 1583.0], [1936.0, 1583.0], [1936.0, 1625.0], [380.0, 1625.0]], ('Explore open roles across our career areas and discover where your skills can make an impact:.', 0.9838844537734985)], [[[2454.0, 1643.0], [2647.0, 1650.0], [2645.0, 1691.0], [2453.0, 1684
"""