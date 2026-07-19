from fastapi import FastAPI
from process import image_processer
from typing import TypedDict
import PIL
from PIL import Image
from pydantic import BaseModel
import uvicorn

app = FastAPI()

class ImageProcesser(BaseModel):
    image_input: str
    box_threshold: float
    iou_threshold: float
    use_paddleocr: bool
    imgsz: int

class MathProcesser(BaseModel):    
    a: int
    b: int

@app.get("/test")
def fastapi_start():
    return {"success": "everything is working"}

@app.post("/image_process")
def process_image(data: ImageProcesser):
    image_input = data.image_input
    box_threshold = data.box_threshold
    iou_threshold = data.iou_threshold
    use_paddleocr = data.use_paddleocr
    imgsz = data.imgsz
    response = image_processer(image_input, box_threshold, iou_threshold, use_paddleocr, imgsz)
    return response

@app.post("/math")
def cal(data: MathProcesser):
    a = data.a
    b = data.b
    return a + b

