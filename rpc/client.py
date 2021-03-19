#!/usr/bin/env python 
# -*- coding: utf-8 -*-

"""
@Time        : 2021-03-18 15:32
@Author      : heyunfan
@Project     : micro-expression-recognition
@File        : client.py
@Description :
"""
import os
import pickle
from io import BytesIO

import numpy as np
import grpc
from rpc import service_pb2, service_pb2_grpc
from rpc.service_pb2 import PredictRequest, PredictReply



def run():
    model_type = "SMIC"
    filename = "asssx.mov"
    channel = grpc.insecure_channel("localhost:50051")
    stub = service_pb2_grpc.PredictorStub(channel)
    response = stub.Predict(PredictRequest(type=model_type, filename=filename))
    print(response.output)



if __name__ == '__main__':
    run()