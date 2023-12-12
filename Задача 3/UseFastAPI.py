from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline


class Item(BaseModel):
    text: str

app = FastAPI()
pipe = pipeline("translation", model="facebook/wmt19-en-ru")

#text = "Я люблю программную инженерию"

@app.post("/predict/")
def predict(item: Item):
    return pipe(item.text)[0]