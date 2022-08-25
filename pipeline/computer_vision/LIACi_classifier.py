# -*- coding: utf-8 -*-
"""
Created on Sun May 23 23:12:44 2021

@author: marynaw
"""

import os
import numpy as np
import cv2
import onnxruntime as rt
import uuid
import csv
import time


class LIACi_classifier():
    def __init__(self, image=None):
        self.image = image
        
        # labels and model
        MODEL_FILENAME = 'computer_vision/modelzoo/LIACi_classifier/model.onnx'
        LABELS_FILENAME = 'computer_vision/modelzoo/LIACi_classifier/labels.txt'

        self.labels = []
        with open(LABELS_FILENAME, 'rt') as lf:
            for l in lf:
                self.labels.append(l.strip())                  

        # onnxruntime inference 
        self.sess = rt.InferenceSession(MODEL_FILENAME, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

    def classify(self):        
        # Get the input name and shape of the model
        input_name = self.sess.get_inputs()[0].name
        h,w = self.sess.get_inputs()[0].shape[2:]
        blob = cv2.dnn.blobFromImage(image=self.image, size=(w,h), swapRB=False, crop=False)
        start = time.perf_counter()
        #Running the session by passing in the input data of the model
        out = self.sess.run(None, {input_name: blob})
        end = time.perf_counter()
        inference_time = end - start
        # print("onnx classification: "+str(inference_time))
        scores = list(out[1][0].values())
        return scores

    def classify_dict(self, frame):
        self.image = frame
        result = self.classify()
        return {key: value for key, value in zip(self.labels, result)}