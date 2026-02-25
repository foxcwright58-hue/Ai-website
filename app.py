from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

app = FastAPI(title="School AI Website")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

TRAIN_TEXTS = [
    "I love this", "This is amazing", "So happy", "That was great",
    "I hate this", "This is terrible", "So sad", "That was awful"
]

TRAIN_LABELS = [
    "positive", "positive", "positive", "positive",
    "negative", "negative", "negative", "negative"
]

model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", LogisticRegression())
])

model.fit(TRAIN_TEXTS, TRAIN_LABELS)

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/predict")
async def predict(request: Request):
    data = await request.json()
    text = data.get("text", "")

    if not text:
        return JSONResponse({"error": "Please enter text."}, status_code=400)

    prediction = model.predict([text])[0]
    return {"prediction": prediction}
