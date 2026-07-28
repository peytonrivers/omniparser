from rapidocr_onnxruntime import RapidOCR

# Initializes the engine to run locally on your CPU
engine = RapidOCR()

img_url = "/Users/peytonrivers/Desktop/test_screenshot.png"
result = engine(img_url)

print(result)
