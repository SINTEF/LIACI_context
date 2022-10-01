# -*- coding: utf-8 -*-
"""
Created on Sat Sep 15 10:58:34 2021

@author: maryna.waszak@sintef.no
"""

from itertools import count
import os
import sys
import numpy as np
import cv2
import time
import math
import onnxruntime as rt
from PIL import ImageColor
from PIL import ImageOps, Image
from functools import reduce

def blendImages(background, foreground):

    output = np.zeros_like(background) 
    # normalize alpha channels from 0-255 to 0-1
    alpha_background = background[:,:,3] / 255.0
    alpha_foreground = foreground[:,:,3] / 255.0

    # set adjusted colors
    for color in range(0, 3):
        output[:,:,color] = alpha_foreground * foreground[:,:,color] + \
            alpha_background * background[:,:,color] * (1 - alpha_foreground)

    # set adjusted alpha and denormalize back to 0-255
    output[:,:,3] = (1 - (1 - alpha_foreground) * (1 - alpha_background)) * 255
    return output


class LIACi_segmenter():
    """Class for semantic segmentation
    """
    CONF_THRESHOLD = 0.5  # Confidence threshold
    MASK_THRESHOLD = 0.5  # Mask threshold

    def __init__(self, image=None,  prob_threshold=0.15, max_detections = 20):
        """Initialize the class
        """

        # labels and model
        MODEL_FILENAME = 'computer_vision/modelzoo/LIACi_segmenter/unet_mobilenetv2_10.onnx'
        LABELS_FILENAME = 'computer_vision/modelzoo/LIACi_segmenter/labels_unet_mobilenetv2_10.txt'

        self.prob_threshold = prob_threshold
        self.max_detections = max_detections

        self.image = image

        self.labels = []
        with open(LABELS_FILENAME, 'rt') as lf:
            for l in lf:
                self.labels.append(l.strip())   

        #self.COLORS = np.random.randint(0, 255, size=(len(self.labels), 3),dtype="uint8")
        self.COLORS =  [    
                        ImageColor.getrgb("cyan"),   
                        ImageColor.getrgb("orange"),
                        ImageColor.getrgb("yellow"),
                        ImageColor.getrgb("pink"),
                        ImageColor.getrgb("green"),
                        ImageColor.getrgb("turquoise"),
                        ImageColor.getrgb("red"),
                        ImageColor.getrgb("purple"),
                        ImageColor.getrgb("white"),                       
                        ImageColor.getrgb("blue")                      
                        ]
 
        # onnxruntime inference 
        self.sess = rt.InferenceSession(MODEL_FILENAME, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

    def get_color_for_label(self, label):
        label_to_color = {l: i for i, l in enumerate(self.labels)}
        return self.COLORS[label_to_color[label]]
    

    # Draw the predicted bounding box, colorize and show the mask on the image
    def draw(self, boxes, masks, scores, labels, putText=False):
        background = cv2.cvtColor(np.array(self.image), cv2.COLOR_RGB2RGBA) #np.array(ImageOps.grayscale(Image.fromarray(self.image)).convert("RGB")) 
        masked_image = np.zeros_like(background) #np.array(ImageOps.grayscale(Image.fromarray(self.image)).convert("RGB"))
        masked_image_other = np.zeros_like(background) #np.array(ImageOps.grayscale(Image.fromarray(self.image)).convert("RGB"))
        legend_shift = 0
        for i in reversed(range(len(masks))):
            mask = masks[i]
            color = self.COLORS[i]

            # Extract mask
            r = np.zeros_like(mask).astype(np.uint8)
            g = np.zeros_like(mask).astype(np.uint8)
            b = np.zeros_like(mask).astype(np.uint8)
            a = 148
            r[mask == 1], g[mask == 1], b[mask == 1] = color
            bgr_color = (color[2],color[1],color[0],a)
            bgr_mask = np.stack([b, g, r], axis=2)
            # sea_chest_grating
            # over_board_valve
            # defect
            # corrosion
            # paint_peel
            # propeller
            # anode
            # bilge_keel
            # marine_growth
            # ship_hull
            if self.labels[i] == "corrosion" or self.labels[i] == "defect" or self.labels[i] == "marine_growth" or self.labels[i] == "paint_peel":
                #masked_image = cv2.addWeighted(masked_image, 0.5, bgr_mask, 1, 0)
                masked_image[mask == 1] = bgr_color
            else:
                #masked_image_other = cv2.addWeighted(masked_image_other, 0.5, bgr_mask, 1, 0)
                masked_image_other[mask == 1] = bgr_color
            # plot legend
            if putText: 
                cv2.rectangle(masked_image, (10, 20+legend_shift), (10+20, 20+legend_shift+10), bgr_color, -1)
                cv2.putText(masked_image, self.labels[i], (35, 30+legend_shift), cv2.FONT_HERSHEY_SIMPLEX, 0.7, bgr_color, 2)
            legend_shift = legend_shift+20

            if boxes is not None:
                box = boxes[i]
                # Extract bounding box
                x0 = int(box[0][0])
                y0 = int(box[0][1])
                x1 = int(box[1][0])
                y1 = int(box[1][1])          
                cv2.rectangle(self.image, (x0,y0), (x1,y1), color, thickness=2)
            if scores is not None:
                text = "{}: {:.4f}".format(self.labels[i], scores[i])
                cv2.putText(self.image, text, (x0, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness=2)


        self.image_inspc = blendImages(background, masked_image)# cv2.addWeighted(bw_image, 0.7, masked_image, 0.7, 0)
        self.image_class = blendImages(background, masked_image_other)#cv2.addWeighted(bw_image, 0.7, masked_image_other, 0.7, 0)

            
    
    
    def getPercentage(self, masksA, masksB):
        ''' calculates the the percentage that maskB is
            covering of maskA
        '''
        if len(masksA)==0 or len(masksB)==0:
            return 0
            
        maskA = masksA[0]
        maskB = masksB[0]

        for mask in masksA[1:]:
            maskA = cv2.bitwise_or(maskA,mask)
        for mask in masksB[1:]:
            maskB = cv2.bitwise_or(maskB,mask)

        intersection = cv2.bitwise_and(maskA,maskB)
        area_intersect = cv2.countNonZero(intersection)
        areaA = cv2.countNonZero(maskA)
        percentage = area_intersect/areaA*100.0

        return percentage

    def get_coverages(self, frame = None):
        if frame is not None:
            self.image = frame
        # Get the input name of the model
        input_name = self.sess.get_inputs()[0].name
        h,w = self.image.shape[:2]    
        h_n, w_n = (256, 320)
        blob = cv2.dnn.blobFromImage(self.image, 1/255.,(w_n, h_n), swapRB=True)
        start = time.perf_counter()
        blob = np.transpose(blob, (0, 2, 3, 1))
        #plt.imshow(blob[0,:,:,:])
        #Running the session by passing in the input data of the model
        out = self.sess.run(None, {input_name: blob})
        end = time.perf_counter()
        inference_time = end - start
        # print("onnx segmentation:  "+str(inference_time))
        
        # postprocess onnx output
        masks = np.squeeze(out)
        labels = range(masks.shape[2])
        counts = {self.labels[i]: (np.count_nonzero(masks[:,:,i] > self.MASK_THRESHOLD) / (h_n * w_n)) for i in labels}
        return counts

    def segment_unet(self, frame = None, visualize = False):
        if frame is not None:
            self.image = frame
        # Get the input name of the model
        input_name = self.sess.get_inputs()[0].name
        h,w = self.image.shape[:2]    
        h_n, w_n = (256, 320)
        blob = cv2.dnn.blobFromImage(self.image, 1/255.,(w_n, h_n), swapRB=True)
        start = time.perf_counter()
        blob = np.transpose(blob, (0, 2, 3, 1))
        #plt.imshow(blob[0,:,:,:])
        #Running the session by passing in the input data of the model
        out = self.sess.run(None, {input_name: blob})
        end = time.perf_counter()
        inference_time = end - start
        # print("onnx segmentation:  "+str(inference_time))
        
        # postprocess onnx output
        masks = np.squeeze(out)
        labels = range(masks.shape[2])

        masks_full_res = []
        for i in labels:
            mask_th = masks[:,:,i]>self.MASK_THRESHOLD
            h, w = self.image.shape[:2]
            mask_th = cv2.resize(mask_th.astype(float), dsize=(w,h)) 
            mask_th = np.array(mask_th)>0
            masks_full_res.append(mask_th)

        # Extract the bounding box and mask for each of the detected objects
        if visualize:
            self.draw(None, masks_full_res, None, labels)

        return [
            {'probability': None,
                 'tagId': int(labels[i]),
                 'tagName': self.labels[labels[i]],
                 'boundingBox': {},
                 'mask':{
                        'height': self.image.shape[0],
                        'width': self.image.shape[1],
                        'mask_array': np.array(masks_full_res[i]).ravel()
                }} for i in labels]   

    def segment_mark_rcnn(self):   
        # Get the input name of the model
        input_name = self.sess.get_inputs()[0].name
        h,w = self.image.shape[:2]    
        h_n, w_n = (512,512)
        img_ratio = (h/h_n, w/w_n)

        blob = cv2.dnn.blobFromImage(self.image, 1/255.,(w_n, h_n),mean=[103, 116, 123])
        start = time.time()

        #Running the session by passing in the input data of the model
        out = self.sess.run(None, {input_name: blob})
        end = time.time()
        inference_time = end - start
        # print("onnx segmentation:  "+str(inference_time))
        
        # postprocess onnx output
        scores = list(out[2])
        if len(scores) == 0 or max(scores) <=self.CONF_THRESHOLD:
            return []
        pred_t = [scores.index(x) for x in scores if x>self.CONF_THRESHOLD][-1]
        masks = np.squeeze(out[3]>self.MASK_THRESHOLD, axis=1)
        h, w = self.image.shape[:2]
        masks = [cv2.resize(masks[i,:].astype(float), dsize=(w,h)) for i in range(len(scores))]
        masks = np.array(masks)>0
        labels = list(out[1])
        boxes = [[(i[0]*img_ratio[1], i[1]*img_ratio[0]), (i[2]*img_ratio[1], i[3]*img_ratio[0])] for i in list(out[0])]

        masks = masks[:pred_t+1]
        boxes = boxes[:pred_t+1]
        labels = labels[:pred_t+1]
        scores = scores[:pred_t+1]

        # Extract the bounding box and mask for each of the detected objects
        self.draw(boxes, masks, scores, labels)

        return [
            {'probability': round(float(scores[i]), 8),
                 'tagId': int(labels[i]),
                 'tagName': self.labels[labels[i]],
                 'boundingBox': {
                     'left': math.ceil(round(float(boxes[i][0][0]), 8)),
                     'top': math.ceil(round(float(boxes[i][0][1]), 8)),
                     'right': math.ceil(round(float(boxes[i][1][0]), 8)),
                     'bottom': math.ceil(round(float(boxes[i][1][1]), 8))
                 },
                 'mask':{
                        'height': self.image.shape[0],
                        'width': self.image.shape[1],
                        'mask_array': np.array(masks[i]).ravel()
                }} for i in range(len(boxes))]      

def write_labels_to_video(video_url):
    cap = cv2.VideoCapture(video_url)
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    video_frame = cap.read()[1]
    FRAME_SIZE = (int(video_frame.shape[1]), int(video_frame.shape[0]))
    FPS = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(video_url[:-4]+'_labeled.mp4', fourcc, FPS, FRAME_SIZE)
    TOTAL_NUM_OF_FRAMES = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    my_segmenter = LIACi_segmenter()

    for i in range(2000, 3000, 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        frame = cap.read()[1]
        my_segmenter.image = frame
        outputs = my_segmenter.segment_unet()
        #frame = cv2.resize(frame, NEW_FRAME_SIZE)
        #cv2.imshow("Image", my_segmenter.image)
        #cv2.waitKey(0) 
        out.write(my_segmenter.image)
            
        print('write label frame: '+str(i))

    out.release()
    cap.release()

if __name__ == '__main__':  
    my_segmenter = LIACi_segmenter()

    # Create a list of labels.
    IMAGE_FILENAME = sys.argv[1]
    image_in = cv2.imread(IMAGE_FILENAME)            
    outputs = my_segmenter.segment_unet(frame=image_in, visualize=True)
    print(f"RGB values for labels:")
    for l, rgb in zip(my_segmenter.labels, my_segmenter.COLORS):
        print(f"  {l:<20}: {rgb}")
    # show the output image
    #cv2.imshow("Segmented by class", my_segmenter.image_class)
    # cv2.imshow("Segmented by inpsection criteria", my_segmenter.image_inspc)
    # cv2.waitKey(0)

    if len(sys.argv) > 2 and '-s' == sys.argv[2]:
        cv2.imwrite(f"{sys.argv[3]}/{os.path.basename(IMAGE_FILENAME)}", image_in)
        cv2.imwrite(f"{sys.argv[3]}/{os.path.basename(IMAGE_FILENAME)}_segmented_classes.png", my_segmenter.image_class)
        cv2.imwrite(f"{sys.argv[3]}/{os.path.basename(IMAGE_FILENAME)}_segmented_inspcrit.png", my_segmenter.image_inspc)

    print('Segmentation is DONE')

