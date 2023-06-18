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

# Display a radio button to choose the input type
input_type = st.radio("Select input type", ("Image URL", "File Upload"))

if input_type == "Image URL":
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

        # Create a list to store the cropped images
        cropped_images = []

        # Display the detected objects and their bounding boxes
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [round(i, 2) for i in box.tolist()]
            object_name = model.config.id2label[label.item()]
            st.write(
                f"Detected {object_name} with confidence {round(score.item(), 3)} at location {box}"
            )

            # Crop the detected object from the image
            cropped_image = image.crop(box)
            cropped_images.append(cropped_image)

        # Display the cropped images in a grid layout
        num_cols = 3
        num_images = len(cropped_images)
        num_rows = (num_images + num_cols - 1) // num_cols

        cols = st.columns(num_cols)

        for i in range(num_rows):
            for j in range(num_cols):
                idx = i * num_cols + j
                if idx < num_images:
                    cols[j].image(cropped_images[idx], caption=object_name, use_column_width=True)

        # Display the input image
        st.image(image, caption="Input Image", use_column_width=True)

else:
    # Display an input file uploader for the image
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Load the uploaded image
        image = Image.open(uploaded_file)

        # Process the image using the YOLO model
        inputs = image_processor(images=image, return_tensors="pt")
        outputs = model(**inputs)

        # Predict bounding boxes and classes
        logits = outputs.logits
        bboxes = outputs.pred_boxes

        # Post-process the object detection results
        target_sizes = torch.tensor([image.size[::-1]])
        results = image_processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)[0]

        # Create a list to store the cropped images
        cropped_images = []

        # Display the detected objects and their bounding boxes
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [round(i, 2) for i in box.tolist()]
            object_name = model.config.id2label[label.item()]
            st.write(
                f"Detected {object_name} with confidence {round(score.item(), 3)} at location {box}"
            )

            # Crop the detected object from the image
            cropped_image = image.crop(box)
            cropped_images.append(cropped_image)

        # Display the cropped images in a grid layout
        num_cols = 3
        num_images = len(cropped_images)
        num_rows = (num_images + num_cols - 1) // num_cols

        cols = st.columns(num_cols)

        for i in range(num_rows):
            for j in range(num_cols):
                idx = i * num_cols + j
                if idx < num_images:
                    cols[j].image(cropped_images[idx], caption=object_name, use_column_width=True)

        # Display the input image
        st.image(image, caption="Input Image", use_column_width=True)
