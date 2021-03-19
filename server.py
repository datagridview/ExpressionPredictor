#!/usr/bin/env python 
# -*- coding: utf-8 -*-

"""
@Time        : 2021-03-18 16:48
@Author      : heyunfan
@Project     : micro-expression-recognition
@File        : main.py
@Description :
"""

import uvicorn
from fastapi import FastAPI

from micro_predictor.model import MicroExpSTCNN_with_CASME, MicroExpSTCNN_with_SMIC, init_model
from macro_predictor.macro import predict

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    global modelCASME
    global modelSMIC
    modelCASME = init_model("CASME")
    modelSMIC = init_model("SMIC")


@app.get("/microexpr")
async def prediction(filename: str, type: str):
    if type == "CASME":
        output = MicroExpSTCNN_with_CASME(modelCASME, filename)
    else:
        output = MicroExpSTCNN_with_SMIC(modelSMIC, filename)
    return {"code": 10000, "data": output}

@app.get("/macroexpr")
async def prediction(filename: str):
    output = predict(filename)
    return {"code": 10000, "data": output}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=50003)
