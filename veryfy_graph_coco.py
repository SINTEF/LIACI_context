from pycocotools import mask
import numpy as np
import cv2

from data.datastore import liaci_graph, neo4j_transaction


def remove_black_borders(image):
    y_nonzero, x_nonzero, _ = np.nonzero(image)
    return image[np.min(y_nonzero):np.max(y_nonzero), np.min(x_nonzero):np.max(x_nonzero)]

with neo4j_transaction() as tx:
    mid = 'm320432.0.0'
    query = "MATCH (m:Mosaic{id:$mid}) WITH m MATCH (m:Mosaic) <-[r:IN_MOSAIC]- (f:Image) return m, r, f"
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
            seg_mask = mask.decode(detlist).astype('bool')
            print(seg_mask.shape)
            seg_mask = np.dstack([seg_mask]*3)
            seg[seg_mask > 0] = 255
        H = r['homography']
        H = [H[:3], H[3:6], H[6:]] 
        #H = [H[6:], H[3:6],  H[:3]] 
        H = np.array(H)
        frame_cur = cv2.imread(f'./assets/thumb/{f["id"]}.jpg')
        frame_cur = cv2.resize(frame_cur,(stiched_image.shape[1]//5, stiched_image.shape[0]//5))
        warped_img = cv2.warpPerspective(
            frame_cur, H, (stiched_image.shape[1], stiched_image.shape[0]), flags=cv2.INTER_LINEAR)

        stiched_image[warped_img > 0] = warped_img[warped_img > 0]

    
    stiched_image /= 255.
    cv2.imshow("Stiched image", remove_black_borders(stiched_image))
    seg_result = cv2.addWeighted(stiched_image,0.8,seg/255,0.4,0)
    cv2.imshow("Segmentation Result", remove_black_borders(seg_result))
    cv2.imshow("Original image", remove_black_borders(cv2.imread(f"./assets/mosaics/{mid}.jpg")))
    cv2.waitKey(0)