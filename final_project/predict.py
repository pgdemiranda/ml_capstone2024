import pickle
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd

with open('./final_model.pkl', 'rb') as f:
    model, preprocessor = pickle.load(f)

app = FastAPI()


class InputData(BaseModel):
    codigo_fipe: int
    ano_modelo: int
    ano_referencia: int
    mes_referencia: int
    marca_freq_encoded: float
    classificacao_marca_economical: bool
    classificacao_marca_affordable: bool
    classificacao_marca_luxury: bool
    classificacao_marca_mid_range: bool
    classificacao_marca_super_luxury: bool

@app.post('/predict')
async def predict(data: InputData):
    input_data = pd.DataFrame([data.dict()])

    prediction = model.predict(input_data)

    return {'prediction': float(prediction[0])}