from cv2 import threshold
from data.access.datastore import EntryDoesExistExeption, find_classification_node, find_node, liaci_graph, neo4j_transaction
from data.inspection.image_node import ImageNode, MosaicNode
from data.vismodel.LiShip import LiShip
from data.inspection.LiInspection import LiInspection
from py2neo.matching import NodeMatcher


import py2neo

def merge_node(mosaic: py2neo.Node):
    with neo4j_transaction() as tx:
        tx.graph.push(mosaic)

def merge_nodes(mosaics):
    with neo4j_transaction() as tx:
        for mosaic in mosaics:
            tx.graph.push(mosaic)
        
def delete_mosaic(mosaic_node: py2neo.Node):
    with neo4j_transaction() as tx:
        tx.delete(mosaic_node)
def create_mosaic(mosaic: MosaicNode):
    mosaic_node = find_node(mosaic.label, id = mosaic.id)
    if not mosaic_node:
        with neo4j_transaction() as tx:
            mosaic_node = py2neo.Node(mosaic.label, **mosaic.__dict__)
            tx.create(mosaic_node)
    return mosaic_node

def create(frame: ImageNode, inspection_node, classification_threshold=0.9):

    inspection_node = find_node(LiInspection().label, id=frame.inspection_id)
    if not inspection_node:
        raise ValueError(f"No or multiple inspections found for id {frame.inspection_id}")
        
    classlabel_to_vis = {
        'anode': '102.2',
        'over_board_valve': '631.1',
        'propeller': '413',
        'sea_chest_grating': '632.332',
        'bilge_keel': '465'
    }

    """
    @TODO:
        Classification nodes dont have imo numbers anymore, this is due to the fact, that we want to
        remove redundant information from the KG and use the relationships. Need to create new method
        to query for the right Classification node base on its relation to a ship with te corresponding
        imo number.
    """
    classlabel_to_node = {
        k: find_classification_node(imo=frame.imo, visCode=v)
        for k, v in classlabel_to_vis.items()
    }

    frame.set_neo4j_properties()
    with neo4j_transaction() as tx:
        frame_node = py2neo.Node(frame.label, **frame.__dict__) 
        tx.create(frame_node)

        inspection_relation = py2neo.Relationship(inspection_node, "HAS_FRAME", frame_node)
        tx.create(inspection_relation)

        for k, v in frame.__dict__.items():
            if k in classlabel_to_vis:
                if v > classification_threshold:
                    results = {
                        'segmentation': getattr(frame, f'{k}_segmentation', 0),
                        'classification': getattr(frame, f'{k}_classification', 0)
                    }
                    classification_relation = py2neo.Relationship(frame_node, "DEPICTS", classlabel_to_node[k], **results)
                    tx.create(classification_relation)

        return frame_node

def add_similarity(frame_node_1, frame_node_2, distance, visual = False):
    with neo4j_transaction() as tx:
        relationtype = "VISUALLY_SIMILAR_TO" if visual else "SIMILAR_TO"
        relation = py2neo.Relationship(frame_node_1, relationtype, frame_node_2, distance=distance)
        tx.create(relation)

def add_homography(frame_node, mosaic_node, homography):
    with neo4j_transaction() as tx:
        relation = py2neo.Relationship(frame_node, "IN_MOSAIC", mosaic_node, homography=[float(x) for l in homography for x in l])
        tx.create(relation)