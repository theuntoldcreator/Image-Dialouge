#Importing the FastAPI
from fastapi import FastAPI, File, UploadFile
#Importing the Fast Api Responses
from fastapi.responses import JSONResponse, RedirectResponse
#Importing transformers
from transformers import ViltProcessor, ViltForQuestionAnswering
#Importing PIL
from PIL import Image
import requests
import io
#Define the FastAPI
app = FastAPI(title="Image-Dialoge")

#Loading the model and tokenizer
vilt_processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
vilt_model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

def get_result(file_image, input_text):
    try:
        # Load and process the file_image
        img = Image.open(io.BytesIO(file_image)).convert("RGB")

        # Prepare inputs
        encoding = vilt_processor(img, input_text, return_tensors="pt")

        # Forward pass
        outputs = vilt_model(**encoding)
        # It shows the index Processing
        logits = outputs.logits
        idx = logits.argmax(-1).item()
        # Shows the model Execution
        result = vilt_model.config.id2label[idx]
#To Return the Answer
        return result

    except Exception as e:
        return str(e)
# Defining the Path
@app.get("/", include_in_schema=False)
async def index():
    return RedirectResponse(url="/docs")
#defining the Path to Show the result
@app.post("/result")
# To Post the Answer
async def process_file_image(file_image: UploadFile = File(...), input_text: str = None):
    try:
        # Using the try to fetch the result
        result = get_result(await file_image.read(), input_text)
        return JSONResponse({"Answer": result})

    except Exception as e:
        #If any Exception arises , it prints this :
        return JSONResponse({"Sorry, Contact the administrator": str(e)})
