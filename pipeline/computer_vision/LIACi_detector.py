# -*- coding: utf-8 -*-
"""
Created on Sat Sep 4 11:27:44 2021

@author: maryna.waszak@sintef.no
"""

import os
import numpy as np
import cv2
import uuid
import math
import onnxruntime as rt
import time

class LIACi_detector():
    """Class for Custom Vision's exported object detection model
    """

    OUTPUT_TENSOR_NAMES = ['detected_boxes', 'detected_scores', 'detected_classes']
    ANCHORS = np.array([[0.573, 0.677], [1.87, 2.06], [3.34, 5.47], [7.88, 3.53], [9.77, 9.17]])
    IOU_THRESHOLD = 0.45
    DEFAULT_INPUT_SIZE = 512 * 512 

    def __init__(self, image=None,  prob_threshold=0.15, max_detections = 20):
        """Initialize the class
        """

        # labels and model
        MODEL_FILENAME = 'computer_vision/modelzoo/LIACi_detector/model.onnx'
        LABELS_FILENAME = 'computer_vision/modelzoo/LIACi_detector/labels.txt'

        self.prob_threshold = prob_threshold
        self.max_detections = max_detections

        self.image = image

        self.img_h = None
        self.img_w = None

        self.labels = []
        with open(LABELS_FILENAME, 'rt') as lf:
            for l in lf:
                self.labels.append(l.strip())   

        self.COLORS = np.random.randint(0, 255, size=(len(self.labels), 3),dtype="uint8")

        # onnxruntime inference 
        self.session = rt.InferenceSession(MODEL_FILENAME, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

    def _logistic(self, x):
        return np.where(x > 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))

    def _non_maximum_suppression(self, boxes, class_probs, max_detections):
        """Remove overlapping bouding boxes
        """
        assert len(boxes) == len(class_probs)

        max_detections = min(max_detections, len(boxes))
        max_probs = np.amax(class_probs, axis=1)
        max_classes = np.argmax(class_probs, axis=1)

        areas = boxes[:, 2] * boxes[:, 3]

        selected_boxes = []
        selected_classes = []
        selected_probs = []

        while len(selected_boxes) < max_detections:
            # Select the prediction with the highest probability.
            i = np.argmax(max_probs)
            if max_probs[i] < self.prob_threshold:
                break

            # Save the selected prediction
            selected_boxes.append(boxes[i])
            selected_classes.append(max_classes[i])
            selected_probs.append(max_probs[i])

            box = boxes[i]
            other_indices = np.concatenate((np.arange(i), np.arange(i + 1, len(boxes))))
            other_boxes = boxes[other_indices]

            # Get overlap between the 'box' and 'other_boxes'
            x1 = np.maximum(box[0], other_boxes[:, 0])
            y1 = np.maximum(box[1], other_boxes[:, 1])
            x2 = np.minimum(box[0] + box[2], other_boxes[:, 0] + other_boxes[:, 2])
            y2 = np.minimum(box[1] + box[3], other_boxes[:, 1] + other_boxes[:, 3])
            w = np.maximum(0, x2 - x1)
            h = np.maximum(0, y2 - y1)

            # Calculate Intersection Over Union (IOU)
            overlap_area = w * h
            iou = overlap_area / (areas[i] + areas[other_indices] - overlap_area)

            # Find the overlapping predictions
            overlapping_indices = other_indices[np.where(iou > self.IOU_THRESHOLD)[0]]
            overlapping_indices = np.append(overlapping_indices, i)

            # Set the probability of overlapping predictions to zero, and udpate max_probs and max_classes.
            class_probs[overlapping_indices, max_classes[i]] = 0
            max_probs[overlapping_indices] = np.amax(class_probs[overlapping_indices], axis=1)
            max_classes[overlapping_indices] = np.argmax(class_probs[overlapping_indices], axis=1)

        assert len(selected_boxes) == len(selected_classes) and len(selected_boxes) == len(selected_probs)
        return selected_boxes, selected_classes, selected_probs

    def _extract_bb(self, prediction_output, anchors):
        assert len(prediction_output.shape) == 3
        num_anchor = anchors.shape[0]
        height, width, channels = prediction_output.shape
        assert channels % num_anchor == 0

        num_class = int(channels / num_anchor) - 5
        assert num_class == len(self.labels)

        outputs = prediction_output.reshape((height, width, num_anchor, -1))

        # Extract bouding box information
        x = (self._logistic(outputs[..., 0]) + np.arange(width)[np.newaxis, :, np.newaxis]) / width
        y = (self._logistic(outputs[..., 1]) + np.arange(height)[:, np.newaxis, np.newaxis]) / height
        w = np.exp(outputs[..., 2]) * anchors[:, 0][np.newaxis, np.newaxis, :] / width
        h = np.exp(outputs[..., 3]) * anchors[:, 1][np.newaxis, np.newaxis, :] / height

        # (x,y) in the network outputs is the center of the bounding box. Convert them to top-left.
        x = x - w / 2
        y = y - h / 2
        boxes = np.stack((x, y, w, h), axis=-1).reshape(-1, 4)

        # Get confidence for the bounding boxes.
        objectness = self._logistic(outputs[..., 4])

        # Get class probabilities for the bounding boxes.
        class_probs = outputs[..., 5:]
        class_probs = np.exp(class_probs - np.amax(class_probs, axis=3)[..., np.newaxis])
        class_probs = class_probs / np.sum(class_probs, axis=3)[..., np.newaxis] * objectness[..., np.newaxis]
        class_probs = class_probs.reshape(-1, num_class)

        assert len(boxes) == len(class_probs)
        return (boxes, class_probs)      

    def postprocess(self, prediction_outputs, input_name):
        """ Extract bounding boxes from the model outputs.

        Args:
            prediction_outputs: Output from the object detection model. (H x W x C)

        Returns:
            List of Prediction objects.
        """
        if input_name=='image_tensor':
            boxes   =  prediction_outputs[0][0]
            class_ids = prediction_outputs[1][0]
            class_probs = prediction_outputs[2].transpose()
        else:
            results = np.squeeze(prediction_outputs).transpose((1,2,0)).astype(np.float32) 
            boxes, class_probs = self._extract_bb(results, self.ANCHORS)

        # Remove bounding boxes whose confidence is lower than the threshold.
        max_probs = np.amax(class_probs, axis=1)
        index, = np.where(max_probs > self.prob_threshold)
        index = index[(-max_probs[index]).argsort()]

        # Remove overlapping bounding boxes
        selected_boxes, selected_classes, selected_probs = self._non_maximum_suppression(boxes[index],
                                                                                         class_probs[index],
                                                                                         self.max_detections)

        
        
        return [
            {'probability': round(float(selected_probs[i]), 8),
                 'tagId': int(selected_classes[i]),
                 'tagName': self.labels[selected_classes[i]],
                 'boundingBox': {
                     'left': math.ceil(round(float(selected_boxes[i][0]), 8)*self.img_w),
                     'top': math.ceil(round(float(selected_boxes[i][1]), 8)*self.img_h),
                     'width': math.ceil(round(float(selected_boxes[i][2]), 8)*self.img_w),
                     'height': math.ceil(round(float(selected_boxes[i][3]), 8)*self.img_h)
                 }
                 } for i in range(len(selected_boxes))]

    def detect(self, frame=None):   
        # Convert to OpenCV format
        if frame is not None:
            self.image = frame
        self.image = cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)
  
        h, w = self.image.shape[:2]
        self.img_h = h
        self.img_w = w
        ratio = math.sqrt(self.DEFAULT_INPUT_SIZE / w / h)
        new_width = int(w * ratio)
        new_height = int(h * ratio)
        new_width = 32 * math.ceil(new_width / 32)
        new_height = 32 * math.ceil(new_height / 32)

        # Resize 
        input_name = self.session.get_inputs()[0].name
        h_n, w_n = self.session.get_inputs()[0].shape[2:]
        img_ratio = (h/h_n, w/w_n)
        self.image = cv2.resize(self.image, (w_n,h_n), interpolation = cv2.INTER_LINEAR)
           
        blob = cv2.dnn.blobFromImage(image=self.image)

        start = time.perf_counter()
        #Running the session by passing in the input data of the model
        outputs = self.session.run(None, {input_name: blob})
        end = time.perf_counter()
        inference_time = end - start
        # nx detection: "+str(inference_time))
              
        scores = self.postprocess(outputs, input_name)
        
        return scores
