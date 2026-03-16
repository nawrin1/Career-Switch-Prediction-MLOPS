from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.responses import HTMLResponse, RedirectResponse
from uvicorn import run as app_run

from typing import Optional
import pandas as pd
from dotenv import load_dotenv
import os
load_dotenv()

# আপনার প্রজেক্টের কনস্ট্যান্ট এবং পাইপলাইন ইমপোর্ট
from src.constants import APP_HOST, APP_PORT
from src.pipline.prediction_pipeline import CareerData, CareerClassifier
from src.pipline.training_pipeline import TrainPipeline

# FastAPI অ্যাপ ইনিশিয়ালাইজ করা
app = FastAPI()

# স্ট্যাটিক ফাইল (CSS) মাউন্ট করা
app.mount("/static", StaticFiles(directory="static"), name="static")

# HTML টেমপ্লেট সেটআপ
templates = Jinja2Templates(directory='templates')

# CORS কনফিগারেশন
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DataForm:
    """
    HTML ফর্ম থেকে ডাটা গ্রহণ করার ক্লাস।
    """
    def __init__(self, request: Request):
        self.request: Request = request
        self.city: Optional[str] = None
        self.city_development_index: Optional[float] = None
        self.gender: Optional[str] = None
        self.relevent_experience: Optional[str] = None
        self.enrolled_university: Optional[str] = None
        self.education_level: Optional[str] = None
        self.major_discipline: Optional[str] = None
        self.experience: Optional[str] = None
        self.company_size: Optional[str] = None
        self.company_type: Optional[str] = None
        self.last_new_job: Optional[str] = None
        self.training_hours: Optional[float] = None

    async def get_career_data(self):
        form = await self.request.form()
        self.city = form.get("city")
        self.city_development_index = form.get("city_development_index")
        self.gender = form.get("gender")
        self.relevent_experience = form.get("relevent_experience")
        self.enrolled_university = form.get("enrolled_university")
        self.education_level = form.get("education_level")
        self.major_discipline = form.get("major_discipline")
        self.experience = form.get("experience")
        self.company_size = form.get("company_size")
        self.company_type = form.get("company_type")
        self.last_new_job = form.get("last_new_job")
        self.training_hours = form.get("training_hours")

# ১. হোম পেজ রাউট (ইন্ডেক্স পেজ রেন্ডার করবে)
@app.get("/", tags=["authentication"])
async def index(request: Request):
    return templates.TemplateResponse(
            "index.html", {"request": request, "context": None})

# ২. ট্রেইনিং রাউট (মডেল ট্রেইন করার জন্য)
@app.get("/train")
async def trainRouteClient():
    try:
        train_pipeline = TrainPipeline()
        train_pipeline.run_pipeline()
        return Response("Training successful!!!")
    except Exception as e:
        return Response(f"Error Occurred! {e}")

# ৩. প্রেডিকশন রাউট (ফর্ম সাবমিট করলে এখানে আসবে)
@app.post("/")
async def predictRouteClient(request: Request):
    try:
        form = DataForm(request)
        await form.get_career_data()
        
        # প্রেডিকশন পাইপলাইনের জন্য ডাটা অবজেক্ট তৈরি
        career_data = CareerData(
            city = form.city,
            city_development_index = form.city_development_index,
            gender = form.gender,
            relevent_experience = form.relevent_experience,
            enrolled_university = form.enrolled_university,
            education_level = form.education_level,
            major_discipline = form.major_discipline,
            experience = form.experience,
            company_type = form.company_type,
            last_new_job = form.last_new_job,
            training_hours = form.training_hours,
            company_size_max = form.company_size # আমরা কোম্পানি সাইজ ড্রপডাউন থেকে নিচ্ছি
        )

        # ইনপুট ডাটাকে ডাটাফ্রেমে রূপান্তর
        career_df = career_data.get_career_input_data_frame()

        # ক্লাসিফায়ার কল করা
        model_predictor = CareerClassifier()

        # প্রেডিকশন নেওয়া
        # মডেল ১ দিলে 'Looking for change', ০ দিলে 'Not looking for change'
        value = model_predictor.predict(dataframe=career_df)[0]

        status = "Employee is looking for a career switch!" if value == 1 else "Employee is NOT looking for a career switch."

        return templates.TemplateResponse(
            "index.html",
            {"request": request, "context": status},
        )
        
    except Exception as e:
        return {"status": False, "error": f"{e}"}

# সার্ভার স্টার্ট করা
if __name__ == "__main__":
    app_run(app, host=APP_HOST, port=APP_PORT)