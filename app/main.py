from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse, JSONResponse
from PIL import Image, ImageOps, ImageEnhance
import numpy as np
import cv2
from io import BytesIO
import base64

app = FastAPI()

# --- Deskew using OpenCV ---
def deskew_image_strict(pil_img: Image.Image) -> Image.Image:
    gray = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2GRAY)
    gray = cv2.bitwise_not(gray)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]

    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    (h, w) = gray.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(np.array(pil_img), M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return Image.fromarray(rotated)

# --- Enhance with Pillow ---
def enhance_image(image: Image.Image) -> Image.Image:
    gray = ImageOps.grayscale(image)
    if gray.width < 1200:
        gray = gray.resize((int(gray.width * 1.5), int(gray.height * 1.5)), Image.BICUBIC)
    contrast = ImageEnhance.Contrast(gray).enhance(1.2)
    sharpened = ImageEnhance.Sharpness(contrast).enhance(1.5)
    return sharpened.convert("RGB")

# --- Auto Crop ---
def autocrop(pil_img: Image.Image) -> Image.Image:
    img_array = np.array(pil_img)
    if img_array.ndim == 2:
        mask = img_array < 250
    else:
        mask = np.mean(img_array, axis=2) < 250
    coords = np.argwhere(mask)
    if coords.size > 0:
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1
        return pil_img.crop((x0, y0, x1, y1))
    return pil_img

# --- Base64 Outputs ---
def prepare_outputs(image: Image.Image):
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    img_bytes = buffer.getvalue()
    base64_img = base64.b64encode(img_bytes).decode("utf-8")
    return {
        "image_base64": base64_img,
        "mime_type": "image/jpeg",
        "file_name": "enhanced.jpg"
    }

def prepare_pdf_output(image: Image.Image):
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="PDF")
    pdf_bytes = buffer.getvalue()
    base64_pdf = base64.b64encode(pdf_bytes).decode("utf-8")
    return {
        "pdf_base64": base64_pdf,
        "mime_type": "application/pdf",
        "file_name": "enhanced_output.pdf"
    }

# --- API Endpoint ---
@app.post("/process-image")
async def process_image(file: UploadFile = File(...)):
    try:
        # ✅ EXIF-safe loading
        image_data = await file.read()
        image = Image.open(BytesIO(image_data))
        image = ImageOps.exif_transpose(image)
        image = image.convert("RGB")

        aligned = deskew_image_strict(image)
        enhanced = enhance_image(aligned)
        cropped = autocrop(enhanced)

        image_result = prepare_outputs(cropped)
        pdf_result = prepare_pdf_output(cropped)

        return JSONResponse(content={
            "image_result": image_result,
            "pdf_result": pdf_result
        })

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

