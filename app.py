import streamlit as st
from PIL import Image
import torch
import requests
import os
import base64
from transformers import YolosImageProcessor, YolosForObjectDetection

# Load the YOLO model and image processor
model = YolosForObjectDetection.from_pretrained('hustvl/yolos-tiny')
image_processor = YolosImageProcessor.from_pretrained("hustvl/yolos-tiny")

# Function to crop image using coordinates
def crop_image(image_url, coordinates):
    # Download the image and save it locally
    image_path = os.path.basename(image_url)
    response = requests.get(image_url, stream=True)
    response.raise_for_status()
    with open(image_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    # Open the downloaded image
    image = Image.open(image_path)

    # Crop the image using the coordinates
    cropped_image = image.crop(coordinates)

    # Remove the downloaded image
    os.remove(image_path)

    return cropped_image

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

    # Create a list to store cropped images and their filenames
    cropped_images = []
    cropped_filenames = []

    # Display the detected objects and their bounding boxes
    for idx, (score, label, box) in enumerate(zip(results["scores"], results["labels"], results["boxes"])):
        box = [round(i, 2) for i in box.tolist()]
        st.write(
            f"Detected {model.config.id2label[label.item()]} with confidence "
            f"{round(score.item(), 3)} at location {box}"
        )
        
        # Crop the image based on the bounding box coordinates
        cropped_image = crop_image(url, box)
        cropped_images.append(cropped_image)

        # Generate a unique filename for the cropped image
        filename = f"cropped_{label.item()}_{idx}.jpg"
        cropped_filenames.append(filename)

        # Save the cropped image in JPEG format
        cropped_image.convert('RGB').save(filename, format='JPEG')

    # Display the input image
    st.image(image, caption="Input Image", use_column_width=True)

    # Arrange cropped images in a matrix with download buttons
    num_images = len(cropped_images)
    num_columns = 3
    num_rows = (num_images + num_columns - 1) // num_columns

    for row in range(num_rows):
        cols = st.columns(num_columns)
        for col in cols:
            if cropped_images:
                cropped_image = cropped_images.pop(0)
                filename = cropped_filenames.pop(0)

                # Display the cropped image
                col.image(cropped_image, caption=f"Cropped Image", use_column_width=True)

                # Read the cropped image data
                with open(filename, "rb") as f:
                    image_data = f.read()

                # Generate base64-encoded image data
                b64_image = base64.b64encode(image_data).decode()

                # Add a download button for the cropped image
                download_button_str = f"Download {filename}"
                col.markdown(f'<a href="data:image/jpeg;base64,{b64_image}" download="{filename}"><button type="button">{download_button_str}</button></a>', unsafe_allow_html=True)
