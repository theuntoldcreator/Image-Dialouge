#Importing streamstlit
import streamlit as streamst
#Importing PIL
from PIL import Image
#Importing BytesIO
from io import BytesIO
#importing transformers , #imporitng VILTProcessor , #importing VILTQA
from transformers import ViltProcessor, ViltForQuestionAnswering

# Initialize the VILT processor and model
vilt_processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
vilt_model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

def get_answer(data_image, question_user):
    try:
        # Load and process the uploaded image
        image = Image.open(BytesIO(data_image)).convert("RGB")

        # Prepare inputs for the VILT model
        encoding_input = vilt_processor(image, question_user, return_tensors="pt")

        # Forward pass through the VILT model

        model_output = vilt_model(**encoding_input)

        logits = model_output.logits
        #Predicting the value
        predicted_index = logits.argmax(-1).item()
        #Starting the execution of the model
        Finalanswer = vilt_model.config.id2label[predicted_index]
# to return answer
        return Finalanswer

    except Exception as e:
        return str(e)

# Set up the streamstlit app
# Added Title
streamst.title("Image-Dialogue")
streamst.write("Upload an image and enter a question to get an answer.")

# Create columns for image upload and input fields
column_image, question_column = streamst.columns(2)

# Image upload
with column_image:
    image_uploaded = streamst.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    if image_uploaded is not None:
        streamst.image(image_uploaded, use_column_width=True)
    else:
        streamst.write("Image Not Yet Uploaded.")

# User question input
with question_column:
    #Enter Your Question
    question_user = streamst.text_input("Enter Your Question")

    # Process the image and question when both are provided
    if image_uploaded and question_user:
        #Added Submit Button
        if streamst.button("Submit"):
            # Process
            data_image = Image.open(image_uploaded)
            image_byte_array = BytesIO()
            # Image Data Is Saved as PNG
            data_image.save(image_byte_array, format='PNG')
            processed_image_bytes = image_byte_array.getvalue()

            # To Get the answer using the VILT model
            generated_answer = get_answer(processed_image_bytes, question_user)

            # TO Display the generated answer
            streamst.success("Answer: " + generated_answer)
