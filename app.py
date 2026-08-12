from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import insightface
import cv2
import numpy as np

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

print("در حال بارگذاری مدل سبک...")
face_app = insightface.app.FaceAnalysis(name='buffalo_s')
face_app.prepare(ctx_id=-1, det_size=(640, 640))
print("مدل آماده شد!")

@app.post("/compare")
async def compare(image1: UploadFile = File(...), image2: UploadFile = File(...)):
    try:
        data1 = await image1.read()
        data2 = await image2.read()
        
        img1 = cv2.imdecode(np.frombuffer(data1, np.uint8), cv2.IMREAD_COLOR)
        img2 = cv2.imdecode(np.frombuffer(data2, np.uint8), cv2.IMREAD_COLOR)
        
        if img1 is None or img2 is None:
            return {"error": "فرمت عکس معتبر نیست", "confidence": 0, "same_person": False}
        
        faces1 = face_app.get(img1)
        faces2 = face_app.get(img2)
        
        if len(faces1) == 0 or len(faces2) == 0:
            return {"error": "چهرهای پیدا نشد", "confidence": 0, "same_person": False}
        
        face1 = max(faces1, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
        face2 = max(faces2, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
        
        emb1 = face1.embedding
        emb2 = face2.embedding
        similarity = float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))
        confidence = round(similarity * 100, 2)
        
        return {
            "confidence": confidence,
            "same_person": bool(confidence > 75)
        }
    
    except Exception as e:
        return {"error": str(e), "confidence": 0, "same_person": False}

@app.get("/")
async def home():
    return {"status": "API فعال است"}
