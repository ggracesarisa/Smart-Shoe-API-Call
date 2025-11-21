from fastapi import FastAPI, UploadFile, File, HTTPException
from google import genai
from PIL import Image
import io
import os
from mangum import Mangum

API_KEY = os.environ.get("GEMINI_API_KEY")

app = FastAPI()

# Initialise Gemini Client
try:
    client = genai.Client(api_key=API_KEY)
except Exception as e:
    print(f"Error initializing Gemini client: {e}")
    client = None

@app.post("/api/analyze-shoe")
async def analyze_shoe(image: UploadFile = File(...)):
    if not client:
        raise HTTPException(status_code=500, detail="Gemini Client is not configured.")
    
    img_bytes = await image.read()
    try:
        img_pil = Image.open(io.BytesIO(img_bytes))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file uploaded.")

    prompt = (
        "You are a public shoe-drying kiosk with a fan + heater. "
        "Assume the temperature is moderate (40–55°C) and drying time "
        "should not exceed 1 hour (60 minutes) to avoid damage. "
        "Suggest a recommended drying time based on shoe type and apparent thickness/material. "
        "Return the output STRICTLY in Thai. "
        "Output format:\n"
        "1) Shoe type with short description (e.g., รองเท้าผ้าใบ (ความหนาปานกลาง))\n"
        "2) Recommended drying time in minutes (e.g., เวลาที่แนะนำ: 40 นาที)\n"
    )

    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[img_pil, prompt]
        )
        result = response.text
        if not result:
            raise Exception("Gemini returned an empty response.")

        return {"status": "success", "result": result}

    except Exception as e:
        print(f"Gemini API Error: {e}")
        raise HTTPException(status_code=500, detail=f"Gemini API failed: {str(e)}")

@app.get("/api")
def root():
    return {"status": "ok", "message": "Shoe Analyzer API is running."}

# สำหรับ Vercel ใช้ Mangum adapter
handler = Mangum(app)
