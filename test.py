from rapidocr_onnxruntime import RapidOCR
import time
from PIL import Image
from process import image_processer
import io
import base64

# Initializes the engine to run locally on your CPU
engine = RapidOCR()

img_url = "/Users/peytonrivers/Desktop/test_screenshot.png"
img = Image.open(img_url)
buffer = io.BytesIO()
new_image = img.save(buffer, format="PNG")
new_bytes = buffer.getvalue()
str_new_bytes = base64.b64encode(new_bytes).decode("utf-8")

start_time = time.perf_counter()

result = engine(img_url)[0]

end_result = time.perf_counter() - start_time

output = image_processer(image_input=str_new_bytes, box_threshold=0.05, iou_threshold=0.10, use_paddleocr=True, imgsz=640)
print(f"nothing has popped up")
time.sleep(10)
encoded_bytes = output["encoded_bytes"]
print(type(encoded_bytes))
formatted_bytes = base64.b64decode(encoded_bytes.encode("utf-8"))
buffer_value = io.BytesIO(formatted_bytes)
final_img = Image.open(buffer_value)
final_img.show()
boxes_details = output["boxes_details"]
print(f"Boxes details: {boxes_details}")

print("Image processer completed")

