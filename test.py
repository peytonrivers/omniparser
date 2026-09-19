from rapidocr_onnxruntime import RapidOCR
import time
from PIL import Image
from process import image_processer
import io
import base64
import numpy as np

# Initializes the engine to run locally on your CPU
"""engine = RapidOCR()

img_url1 = "/Users/peytonrivers/Desktop/small10.png"
img_url2 = "/Users/peytonrivers/Desktop/small11.png"
img1 = Image.open(img_url1)
img1.show()
img2 = Image.open(img_url2)
img2.show()
img_width1, img_height1 = img1.size
img_width2, img_height2 = img2.size
matrix1 = np.array(img1)
matrix2 = np.array(img2)
matrix3 = reversed(matrix1)
matrix3 = list(matrix3)
print(matrix1[0][0])
print(matrix2[0][0])
matrix4 = reversed(matrix2)
matrix4 = list(matrix4)
print(len(matrix1))
print(matrix1.shape)
print(matrix2.shape)

def first(matrix1, matrix2):
    for i in range(len(matrix1)):
        width1 = matrix1[i]
        width2 = matrix2[i]
        for l in range(len(width1)):
            elem1 = width1[l]
            elem2 = width2[l]
            for j in range(len(elem1)):
                num1 = elem1[j]
                num2 = elem2[j]
                if num1 != num2:
                    print(f"Row: {i+1}")
                    print(elem1)
                    print(elem2)
                    return i


result = first(matrix1, matrix2)
reversed_result = first(matrix3, matrix4)
backwards_result = img_width1 - reversed_result
cropped_box2 = (0, result, img_width2, backwards_result)
cropped_image2 = img2.crop(cropped_box2)
cropped_image2.show()"""

"""buffer = io.BytesIO()
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

print("Image processer completed")"""


my_data = {"item": "Peyton", "last": "Rivers"}
for key, value in my_data.items():
    print(f"key: {key}")
    print(f"value: {value}")