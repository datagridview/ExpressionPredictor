#!/usr/bin/env python 
# -*- coding: utf-8 -*-

"""
@Time        : 2021-03-18 14:44
@Author      : heyunfan
@Project     : micro-expression-recognition
@File        : server.py
@Description :
"""
import pickle
import time
from concurrent import futures

import grpc
from rpc import service_pb2_grpc, service_pb2
from CASMES.model import predict, init_model, MicroExpSTCNN_with_CASME, MicroExpSTCNN_with_SMIC

_ONE_DAY_IN_SECONDS = 60 * 60 * 24
MAX_MESSAGE_LENGTH = 256*1024*1024


class Predictor(service_pb2_grpc.PredictorServicer):
    def __init__(self):
        self.modelCASME = init_model("CASME")
        self.modelSMIC = init_model("SMIC")

    def Predict(self, request, context):
        if request.type == "CASME":
            output = MicroExpSTCNN_with_CASME(self.modelCASME, request.filename)
        else:
            output = MicroExpSTCNN_with_SMIC(self.modelSMIC, request.filename)
        return service_pb2.PredictReply(output=output)


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10), options=[
               ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
               ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),
               ])
    service_pb2_grpc.add_PredictorServicer_to_server(Predictor(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == '__main__':
    serve()
