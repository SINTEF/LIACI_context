# -*- coding: utf-8 -*-
"""
Created on Sun May 28 11:33:02 2022

@author: marynaw
"""
from math import prod
import imreg_dft as ird
import numpy as np
import cv2
from computer_vision.LIACi_segmenter import LIACi_segmenter, blendImagesAlpha

from pycocotools import mask
import logging
logger = logging.getLogger('stitcher')

class Mosaic:
    def __init__(self, first_image, output_height_times=5, output_width_times=5, detector_type="sift"):
        """This class processes every frame and generates the panorama

        Args:
            first_image (image for the first frame): first image to initialize the output size
            output_height_times (int, optional): determines the output height based on input image height. Defaults to 2.
            output_width_times (int, optional): determines the output width based on input image width. Defaults to 4.
            detector_type (str, optional): the detector for feature detection. It can be "sift" or "orb". Defaults to "sift".
        """
        self.detector_type = detector_type
        if detector_type == "sift":
            self.detector = cv2.SIFT_create(700)
            self.bf = cv2.BFMatcher()
        elif detector_type == "orb":
            self.detector = cv2.ORB_create(700)
            self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        self.visualize = False

        self.process_first_frame(first_image)

        self.output_img = np.full((int(output_height_times * first_image.shape[0]), int(
            output_width_times*first_image.shape[1]), first_image.shape[2]), 255.)

        # offset
        self.w_offset = int(self.output_img.shape[0]/2 - first_image.shape[0]/2)
        self.h_offset = int(self.output_img.shape[1]/2 - first_image.shape[1]/2)

        self.output_img[self.w_offset:self.w_offset+first_image.shape[0],
                        self.h_offset:self.h_offset+first_image.shape[1], :] = first_image

        self.H_old = np.eye(3)
        self.H_old[0, 2] = self.h_offset
        self.H_old[1, 2] = self.w_offset

    def process_first_frame(self, first_image):
        """processes the first frame for feature detection and description

        Args:
            first_image (cv2 image/np array): first image for feature detection
        """
        self.frame_prev = first_image
        frame_gray_prev = cv2.cvtColor(first_image, cv2.COLOR_BGR2GRAY)
        self.kp_prev, self.des_prev = self.detector.detectAndCompute(frame_gray_prev, None)

    def match_with_phase_corr(self, cur_frame, prev_frame):
        """matches the images through phase correlation

        Args:
            cur_frame (np array): current frame in grey scale
            prev_frame (np array): previous frame in grey scale

        Returns:
            M: transfomation matrix as np array
        """
        try:
            result = ird.similarity(prev_frame, cur_frame, numiter=3)
        except ValueError:
            return None
        rows,cols = cur_frame.shape
        M = cv2.getRotationMatrix2D((cols//2,rows//2),result['angle'],result['scale'])
        M += np.array([[0,0,result['tvec'][1]],[0,0,result['tvec'][0]]])
        M = np.append(M,np.array([[0,0,1]]),0)

        return M

    def match(self, des_cur, des_prev):
        """matches the descriptors

        Args:
            des_cur (np array): current frame descriptor
            des_prev (np arrau): previous frame descriptor

        Returns:
            array: and array of matches between descriptors
        """
        # matching
        if self.detector_type == "sift":
            try:
                matches = []
                if des_prev is None:
                    raise ValueError
                pair_matches = self.bf.knnMatch(des_cur, des_prev, k=2)               
                for m, n in pair_matches:
                    if m.distance < 0.7*n.distance:
                        matches.append(m)
            except ValueError:
                pass
                #print("pair_matches are not valid...")

        elif self.detector_type == "orb":
            matches = self.bf.match(des_cur, des_prev)

        # Sort them in the order of their distance.
        matches = sorted(matches, key=lambda x: x.distance)

        # get the maximum of 20  best matches
        matches = matches[:min(len(matches), 20)]
        # Draw first 10 matches.
        if self.visualize:
            match_img = cv2.drawMatches(self.frame_cur, self.kp_cur, self.frame_prev, self.kp_prev, matches, None,
                                        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            cv2.imshow('matches', match_img)
        return matches

    def process_frame(self, frame_cur):
        """gets an image and processes that image for mosaicing

        Args:
            frame_cur (np array): input of current frame for the mosaicing
        """
        if self.frame_prev is not None:
            frame_gray_past = cv2.cvtColor(self.frame_prev, cv2.COLOR_BGR2GRAY)
        self.frame_cur = frame_cur
        frame_gray_cur = cv2.cvtColor(frame_cur, cv2.COLOR_BGR2GRAY)
        self.kp_cur, self.des_cur = self.detector.detectAndCompute(frame_gray_cur, None)

        self.matches = self.match(self.des_cur, self.des_prev)
        #self.matches = []
        if len(self.matches) < 4:  
            #print("unsufficient features - return")
            #return False         
            self.H = self.match_with_phase_corr(frame_gray_cur, frame_gray_past)
            #print("unsufficient features - doing phase correlation")
        else:
            self.H = self.findHomography(self.kp_cur, self.kp_prev, self.matches)

        # if bad Homography due to lack of features
        # we can use phase correlation
        if self.H is None and frame_gray_past is not None:
            self.H = self.match_with_phase_corr(frame_gray_cur, frame_gray_past)
            #print("bad homography - doing phase correlation")

        if self.H is None:
            return False        

        logger.debug("Could find homography for frame.")
        self.H = np.matmul(self.H_old, self.H)

        # check if the reference point is too far away after transformation
        transformed_corners_old = self.get_transformed_corners(self.frame_cur, self.H_old, self.output_img)
        transformed_corners = self.get_transformed_corners(self.frame_cur, self.H, self.output_img)
        distance = np.linalg.norm(transformed_corners_old - transformed_corners)
        if distance > 100: 
            logger.debug("Distortion too big - stopping!")
            return False
    
        # dont warp to much overall...
        warpiness = 0
        for i in range(4):
            c1 = transformed_corners[0][i]
            c2 = transformed_corners[0][(i+1)%4]

            warpiness += min(
                abs(c1[0] - c2[0]), 
                abs(c1[1] - c2[1])
            )

        if warpiness > 260:
            logger.debug("Overall warpiness too big - stopping!")
            return False 
        
            
      
        self.warp(self.frame_cur, self.H)

        # loop preparation
        self.H_old = self.H
        self.kp_prev = self.kp_cur
        self.des_prev = self.des_cur
        self.frame_prev = self.frame_cur

        return True

    @ staticmethod
    def findHomography(image_1_kp, image_2_kp, matches):
        """gets two matches and calculate the homography between two images

        Args:
            image_1_kp (np array): keypoints of image 1
            image_2_kp (np_array): keypoints of image 2
            matches (np array): matches between keypoints in image 1 and image 2

        Returns:
            np arrat of shape [3,3]: Homography matrix
        """
        # taken from https://github.com/cmcguinness/focusstack/blob/master/FocusStack.py

        image_1_points = np.zeros((len(matches), 1, 2), dtype=np.float32)
        image_2_points = np.zeros((len(matches), 1, 2), dtype=np.float32)
        for i in range(0, len(matches)):
            image_1_points[i] = image_1_kp[matches[i].queryIdx].pt
            image_2_points[i] = image_2_kp[matches[i].trainIdx].pt

        homography, mask = cv2.findHomography(
            image_1_points, image_2_points, cv2.RANSAC, ransacReprojThreshold=2.0)

        return homography

    def warp(self, frame_cur, H):
        """ warps the current frame based of calculated homography H

        Args:
            frame_cur (np array): current frame
            H (np array of shape [3,3]): homography matrix

        Returns:
            np array: image output of mosaicing
        """
        warped_img = cv2.warpPerspective(
            frame_cur, H, (self.output_img.shape[1], self.output_img.shape[0]), flags=cv2.INTER_LINEAR)

        transformed_corners = self.get_transformed_corners(frame_cur, H, self.output_img)
        warped_img = self.draw_border(warped_img, transformed_corners)
        if self.visualize:
            cv2.imshow('warped_img',  warped_img/255.)

        self.output_img[warped_img > 0] = warped_img[warped_img > 0]
        if self.visualize:
            output_temp = np.copy(self.output_img)
            output_temp = self.draw_border(output_temp, transformed_corners, color=(0, 0, 255))
            cv2.imshow('output',  output_temp/255.)

        return self.output_img

    @ staticmethod
    def get_transformed_corners(frame_cur, H, output_img):
        """finds the corner of the current frame after warp

        Args:
            frame_cur (np array): current frame
            H (np array of shape [3,3]): Homography matrix 

        Returns:
            [np array]: a list of 4 corner points after warping
        """
        corner_0 = np.array([0, 0])
        corner_1 = np.array([frame_cur.shape[1], 0])
        corner_2 = np.array([frame_cur.shape[1], frame_cur.shape[0]])
        corner_3 = np.array([0, frame_cur.shape[0]])

        corners = np.array([[corner_0, corner_1, corner_2, corner_3]], dtype=np.float32)
        transformed_corners = cv2.perspectiveTransform(corners, H)
        corners = np.array([[corner_0, corner_1, corner_2, corner_3]])
        transformed_corners = np.array(transformed_corners, dtype=np.int32)
        #mask = np.zeros(shape=(output_img.shape[0], output_img.shape[1], 1))
        #cv2.fillPoly(mask, transformed_corners, color=(1, 0, 0))
        #cv2.imshow('mask', mask)

        return transformed_corners

    def draw_border(self, image, corners, color=(0, 0, 0)):
        """This functions draw rectancle border

        Args:
            image ([type]): current mosaiced output
            corners (np array): list of corner points
            color (tuple, optional): color of the border lines. Defaults to (0, 0, 0).

        Returns:
            np array: the output image with border
        """
        for i in range(corners.shape[1]-1, -1, -1):
            cv2.line(image, tuple(corners[0, i, :]), tuple(
                corners[0, i-1, :]), thickness=5, color=color)
        return image


def remove_black_borders(image):
    color_top_left = image[0,0]
    y_nonzero, x_nonzero, _ = np.where(image != color_top_left)
    return image[np.min(y_nonzero):np.max(y_nonzero), np.min(x_nonzero):np.max(x_nonzero)]

class LIACI_Stitcher():
    """
    Wrapper class for the Mosaicing functionallity.
    """
    def __init__(self, labels=None):
        self.segmenter = LIACi_segmenter()
        self.is_first_frame = True

        self.labels = labels
        if self.labels == None:
            self.labels = ['paint_peel', 'marine_growth', 'corrosion', 'ship_hull', 'sea_chest_grating', 'over_board_valve', ]
    
        self.number_of_frames = 0
        self.starting_point = np.array([0, 0])

    def get_dimensions(self):
        return self.video_mosaic.output_img.shape[0], self.video_mosaic.output_img.shape[1]

    def get_size_increase(self):
        pixels_not_black = np.any(self.video_mosaic.output_img != [255.,255.,255.], axis=-1).sum()
        pixels_init = prod(self.get_dimensions()) / 25
        return pixels_not_black / pixels_init


    def store_mosaic_seg(self, path):
        logger.debug("Storing segmented mosaic image...")
        x_dim, y_dim = self.get_dimensions()
        diagonal = np.zeros((x_dim, y_dim), np.uint8)
        for x, y in np.ndindex(diagonal.shape):
            if (x+ y) % 4 == 0:
                diagonal[x][y] = 1

        output_image = np.array(self.video_mosaic.output_img)
        seg_output_overlay = np.zeros(shape=(x_dim, y_dim, 3))
        if 'ship_hull' in self.masks:
            color = self.segmenter.get_color_for_label('ship_hull', True)
            seg_output_overlay[self.masks['ship_hull'] > 0] = color 
            seg_output_overlay[diagonal == 1] = (0,0,0) 
        for label, mask in self.masks.items():
            if label != "ship_hull":
                seg_output_overlay[mask > 0] = self.segmenter.get_color_for_label(label)
        output_image = blendImagesAlpha(output_image,255,seg_output_overlay,90)

        cv2.imwrite(path, remove_black_borders(output_image))
        logger.debug("Done. stored segmented mosaic image...")

    def store_mosaic(self, path):
        output_image = self.video_mosaic.output_img
        cv2.imwrite(path, output_image)

    def get_coco(self):
        result = {}
        for l in self.labels:
            result[l] = mask.encode(np.asfortranarray(self.masks[l]))
        return result 

    def get_percentages(self):
        pixels_not_black = np.any(self.video_mosaic.output_img != [255.,255.,255.], axis=-1)
        num_pixels = pixels_not_black.sum()
        
        result = {}
        for l in self.labels:
            result[l] = np.count_nonzero(self.masks[l][pixels_not_black]) / num_pixels

        return result 


    def _initialize_masks(self):
        self.masks = {
            l: np.zeros((self.video_mosaic.output_img.shape[0],self.video_mosaic.output_img.shape[1]), np.uint8) 
            for l in self.labels
        }

    def next_frame(self, frame):
        NEW_FRAME_SIZE = (int(frame.shape[1]/4), int(frame.shape[0]/4))       
        frame_cur = cv2.resize(frame, NEW_FRAME_SIZE)  
        self.segmenter.image = frame_cur
        seg_results = self.segmenter.segment_unet()
        if self.is_first_frame:
            logger.debug("Is first frame, creating new Mosaic")
            self.video_mosaic = Mosaic(frame_cur, detector_type="sift")           
            self.is_first_frame = False
            self._initialize_masks()
            self.number_of_frames = 0

        # process each frame
        ret_val = self.video_mosaic.process_frame(frame_cur)
        logger.debug(f"The mosaic has processed the frame with the return value {ret_val}")

        if not ret_val:
            self.is_first_frame = True
            return None
                  
        self.number_of_frames += 1

        current_masks = {l: np.zeros((frame_cur.shape[0],frame_cur.shape[1]), np.uint8) for l in self.labels}

        for value in seg_results:      
            # draw the mask  
            label = value['tagName']
            if label in self.labels:
                mask = np.reshape(value['mask']['mask_array'], (value['mask']['height'],value['mask']['width'])).astype(np.uint8)
                current_masks[label] = cv2.bitwise_or(current_masks[label], mask)
       
        for l in self.labels:
            warped_mask = cv2.warpPerspective(
                current_masks[l], self.video_mosaic.H, (self.video_mosaic.output_img.shape[1], self.video_mosaic.output_img.shape[0]), flags=cv2.INTER_CUBIC)
            self.masks[l] = cv2.bitwise_or(self.masks[l], warped_mask)

       
        return self.video_mosaic.H


    
def demo():
    #video_source = 'C:\\Users\\marynaw\\data\\LIACi\\Videos\\Liaci Hull status\\2022-03-09_09.15.37.mp4'
    video_source = '/home/joseph/liaci/Liaci Hull status/2022-03-09_09.15.37.mp4'
    video_capture = cv2.VideoCapture(video_source)
    START_FRAME = int(video_capture.get(cv2.CAP_PROP_FPS)*(34*60+50))
    NUM_OF_FRAMES = 500

    # there is also an image stitcher in openCV but it is hard to configure
    # images = []
    # for i in range(START_FRAME,START_FRAME+NUM_OF_FRAMES,5):
    #     video_capture.set(cv2.CAP_PROP_POS_FRAMES, i)
    #     # Capture frame-by-frame
    #     _, frame = video_capture.read() 
    #     NEW_FRAME_SIZE = (int(frame.shape[1]/4), int(frame.shape[0]/4))       
    #     frame_cur = cv2.resize(frame, NEW_FRAME_SIZE)  
    #     images.append(frame_cur)

    # stitcher = cv2.Stitcher_create() 
    # (status, stitched) = stitcher.stitch(images)
    # cv2.imshow('matches', stitched)
    # cv2.waitKey(0)

    segmenter = LIACi_segmenter()
    is_first_frame = True
    
    number_of_frames = 0
    starting_point = np.array([0, 0])
    local_start_frame = START_FRAME
    FRAME_STEP = 1
    for i in range(START_FRAME,START_FRAME+NUM_OF_FRAMES,FRAME_STEP):
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, i)
        # Capture frame-by-frame
        _, frame = video_capture.read() 
        NEW_FRAME_SIZE = (int(frame.shape[1]/4), int(frame.shape[0]/4))       
        frame_cur = cv2.resize(frame, NEW_FRAME_SIZE)  
        segmenter.image = frame_cur
        seg_results = segmenter.segment_unet()
        if is_first_frame:
            video_mosaic = Mosaic(frame_cur, detector_type="sift")           
            is_first_frame = False
            local_start_frame = i  
            number_of_frames = 0         
            seg_output_img = np.zeros((video_mosaic.output_img.shape[0],video_mosaic.output_img.shape[1],3), np.uint8)

        # process each frame
        ret_val = video_mosaic.process_frame(frame_cur)

        if number_of_frames > 2:
            number_of_stitched_frames = i-local_start_frame
            # if the mosaic is worth saving
            if False:#number_of_stitched_frames>=10:
                cv2.imwrite("mosaic_{}_{}_d{}.jpg".format(local_start_frame,i,FRAME_STEP), video_mosaic.output_img)
                cv2.imwrite("seg_mosaic_{}_{}_d{}.jpg".format(local_start_frame,i,FRAME_STEP), seg_output_img)
            is_first_frame = True  
            continue  

        if not ret_val:
            number_of_frames += 1
            continue
                  

        frame_cur_seg_mg = np.zeros((frame_cur.shape[0],frame_cur.shape[1]), np.uint8)
        frame_cur_seg_sh = np.zeros((frame_cur.shape[0],frame_cur.shape[1]), np.uint8)

        for value in seg_results:      
            color = segmenter.COLORS[value['tagId']]  
            # draw the mask  
            if value['tagName'] == 'ship_hull':
                mask_sh = np.reshape(value['mask']['mask_array'], (value['mask']['height'],value['mask']['width'])).astype(np.uint8)
                frame_cur_seg_sh = cv2.bitwise_or(frame_cur_seg_mg, mask_sh)
            if value['tagName'] == 'marine_growth':
                mask_mg = np.reshape(value['mask']['mask_array'], (value['mask']['height'],value['mask']['width'])).astype(np.uint8)
                frame_cur_seg_mg = cv2.bitwise_or(frame_cur_seg_mg, mask_mg)           

        #@todo: do not overwrite the previous mask but make an OR        
        warped_seg_img_sh = cv2.warpPerspective(
                frame_cur_seg_sh, video_mosaic.H, (video_mosaic.output_img.shape[1], video_mosaic.output_img.shape[0]), flags=cv2.INTER_CUBIC)
        seg_output_img[warped_seg_img_sh > 0] = segmenter.COLORS[9]
        warped_seg_img_mg = cv2.warpPerspective(
                frame_cur_seg_mg, video_mosaic.H, (video_mosaic.output_img.shape[1], video_mosaic.output_img.shape[0]), flags=cv2.INTER_CUBIC)
        seg_output_img[warped_seg_img_mg > 0] = segmenter.COLORS[4]
       
        showimage = np.array(video_mosaic.output_img)
        transformed_corners = video_mosaic.get_transformed_corners(video_mosaic.frame_cur, video_mosaic.H, video_mosaic.output_img)
        video_mosaic.draw_border(showimage, transformed_corners)
        cv2.imshow('preview',  showimage/255.)
        
        #cv2.imshow('preview',  remove_black_borders(video_mosaic.output_img/255.))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()
       

