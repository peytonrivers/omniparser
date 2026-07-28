from rapidocr_onnxruntime import RapidOCR
import time

# Initializes the engine to run locally on your CPU
engine = RapidOCR()

img_url = "/Users/peytonrivers/Desktop/test_screenshot.png"
result = engine(img_url)

start_time = time.perf_counter()

result = engine(img_url)[0]

end_result = time.perf_counter() - start_time

print(result)
print(f"time it took to get the text boxes coordinates: {end_result}")
print(f"Total amount of boxes: {len(result[0])}")

for i in range(len(result)):
    res = result[i]
    print(f"Res 0: {[res[0]]}")
    print(f"Res 1: {res[1]}")
    print(f"Res 2: {res[2]}")

