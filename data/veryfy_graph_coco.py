from pycocotools import mask
import numpy as np
import cv2

from data.access.datastore import neo4j_transaction

from PIL import ImageColor



def remove_black_borders(image):
    y_nonzero, x_nonzero, _ = np.nonzero(image)
    return image[np.min(y_nonzero):np.max(y_nonzero), np.min(x_nonzero):np.max(x_nonzero)]


LABELS_FILENAME = 'pipeline/computer_vision/modelzoo/LIACi_segmenter/labels_unet_mobilenetv2_10.txt'
labels = []
with open(LABELS_FILENAME, 'rt') as lf:
    for l in lf:
        labels.append(l.strip())   

COLORS =  [    
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
 

def get_color_for_label(label):
    label_to_color = {l: i for i, l in enumerate(labels)}
    return COLORS[label_to_color[label]]

def get_color_map():
    for l in labels:
        print(l, get_color_for_label(l))

if __name__ == "__main__":
    with neo4j_transaction() as tx:
        mid = 'm300.3875.0'
        query = "MATCH (m:Mosaic{id:$mid}) WITH m MATCH (m:Mosaic) <-[r:IN_MOSAIC]- (f:Frame) return m, r, f"
        result = tx.run(query, mid=mid)

        stiched_image = None
        for m, r, f in result:
            if stiched_image is None:
                stiched_image = np.zeros(shape=(m['x_dim'], m['y_dim'], 3))
                seg = np.zeros(shape=(m['x_dim'], m['y_dim'], 3))
                print(m['ship_hull_coco_size'])
                detection ={
                    'size': m['marine_growth_coco_size'],
                    'counts': m['marine_growth_coco']
                }
                detlist = []
                detlist.append(detection)   
                seg_mask = mask.decode(detlist).astype('uint8')
                print(seg_mask.shape)
                seg_mask = np.squeeze(seg_mask, axis=2)
                print(seg_mask.shape)
                print(np.zeros((m['x_dim'], m['y_dim']), np.uint8).shape)
                #seg_mask = np.dstack([seg_mask]*3)
                seg[seg_mask > 0] = get_color_for_label('marine_growth')
            H = r['homography']
            H = [H[:3], H[3:6], H[6:]] 
            #H = [H[6:], H[3:6],  H[:3]] 
            H = np.array(H)
            frame_cur = cv2.imread(f'./data/imgs/frames/{f["id"]}.jpg')
            frame_cur = cv2.resize(frame_cur,(stiched_image.shape[1]//5, stiched_image.shape[0]//5))
            warped_img = cv2.warpPerspective(
                frame_cur, H, (stiched_image.shape[1], stiched_image.shape[0]), flags=cv2.INTER_LINEAR)

            stiched_image[warped_img > 0] = warped_img[warped_img > 0]

    
        stiched_image /= 255.
        cv2.imshow("Stiched image", remove_black_borders(stiched_image))
        seg_result = cv2.addWeighted(stiched_image,1,seg,0.01,0)
        cv2.imshow("Segmentation Result", remove_black_borders(seg_result))
        #cv2.imshow("Original image", remove_black_borders(cv2.imread(f"./assets/mosaics/{mid}.jpg")))
        cv2.waitKey(0)