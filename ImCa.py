# ---- PROJECT FOR IMAGE CAPTIONING BY MACHINE
# USING AI/ML BY THE STREAMLIT LIBRARY FOR
# DEPLOYMENT OF PROJECT ----



#--- Importing libraries ---
import streamlit as st
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
import torch
from PIL import Image
import io

# -- Pretrained model for the captioning of image --



model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

# -- function for the prediction of the caption of image (Single) --


def predict_step(image_file):
    try:
        # Open image file
        image = Image.open(io.BytesIO(image_file.read()))

        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert(mode="RGB")

        # Extract features and generate caption
        pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values.to(device)
        output_ids = model.generate(pixel_values, **gen_kwargs)
        preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        preds = [pred.strip() for pred in preds]

        return preds

    except:
        # Catch any errors that occur during processing
        st.error("Error: Could not process image")
        return []



    # -- function for the prediction of the caption of image (Multiple) --



def predict_step_multile(image_file, num_captions=5):
    try:
        # Open image file
        image = Image.open(io.BytesIO(image_file.read()))

        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert(mode="RGB")

        # Extract features and generate captions
        pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values.to(device)
        output_ids = model.generate(pixel_values, **gen_kwargs)
        preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        preds = [pred.strip() for pred in preds]

        # Return multiple captions
        if num_captions > 0:
            return preds[:num_captions]
        else:
            return preds

    except:
        # Catch any errors that occur during processing
        st.error("Error: Could not process image")
        return []




# --Main function for using both function--




def main():



    st.title("Internship project")
    html_temp = """
       <div style="background-color:#025246 ;padding:10px">
       <h2 style="color:white;text-align:center;">Caption making from image app </h2>
       </div>
       """
    st.markdown(html_temp, unsafe_allow_html=True)

    # Allow user to upload an image file
    img_file = st.file_uploader("Upload an image", type=["png","jpg","jpeg"])

    if img_file is not None:


        st.image(img_file, use_column_width=True)

        if st.button("Generate Caption"):
            output = predict_step(img_file)

            st.success(f"The caption of your image: {output}")
        if st.button("generate multiple caption"):
            num_captions = st.slider("Number of Captions", min_value=1, max_value=10, value=5, step=1)
            output = predict_step_multile(img_file, num_captions)

            if output:
                st.success("Captions:")
                for i, caption in enumerate(output):
                    st.write(f"{i + 1}. {caption}")
            else:
                st.info("No captions generated for the image.")
main()