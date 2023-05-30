import streamlit as st
from PIL import Image
import torch
import requests
from transformers import YolosImageProcessor, YolosForObjectDetection

# Load the YOLO model and image processor
model = YolosForObjectDetection.from_pretrained('hustvl/yolos-tiny')
image_processor = YolosImageProcessor.from_pretrained("hustvl/yolos-tiny")

# Streamlit app
st.title("Object Detection with YOLO")

# Display an input text box for the image URL
url = st.text_input("Enter the image URL:")
if url:
    # Load the image from the provided URL
    image = Image.open(requests.get(url, stream=True).raw)

    # Process the image using the YOLO model
    inputs = image_processor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    # Predict bounding boxes and classes
    logits = outputs.logits
    bboxes = outputs.pred_boxes

    # Post-process the object detection results
    target_sizes = torch.tensor([image.size[::-1]])
    results = image_processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)[0]

    # Display the detected objects and their bounding boxes
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        st.write(
            f"Detected {model.config.id2label[label.item()]} with confidence "
            f"{round(score.item(), 3)} at location {box}"
        )

    # Display the input image
    st.image(image, caption="Input Image", use_column_width=True)
