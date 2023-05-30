import streamlit as st
import torch
from PIL import Image
import requests
from transformers import DetrImageProcessor, DetrForObjectDetection


def main():
    st.title("Object Detection with DETR")

    url = st.text_input("Enter the image URL:")
    if url:
        image = Image.open(requests.get(url, stream=True).raw)

        processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

        inputs = processor(images=image, return_tensors="pt")
        outputs = model(**inputs)

        target_sizes = torch.tensor([image.size[::-1]])
        results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [round(i, 2) for i in box.tolist()]
            st.write(
                f"Detected {model.config.id2label[label.item()]} with confidence "
                f"{round(score.item(), 3)} at location {box}"
            )

        st.image(image, caption="Input Image", use_column_width=True)


if __name__ == "__main__":
    main()
