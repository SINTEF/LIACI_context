from cv2 import threshold
from data.datastore import EntryDoesExistExeption, find_node, liaci_graph, neo4j_transaction
from data.inspection.image_node import ImageNode
from data.vismodel.LiShip import LiShip
from data.inspection.LiInspection import LiInspection
from py2neo.matching import NodeMatcher


import py2neo

def merge(frame: ImageNode, **kwargs):
    frame_node = find_node(frame.label, id=frame.id)
    if not frame_node:
        create(frame, **kwargs)
    with neo4j_transaction() as tx:
        frame_node = py2neo.Node(frame.label, **frame.__dict__)
        tx.merge(frame_node)

def create(frame: ImageNode, fail_on_exists = False, classification_threshold=0.9):
    frame_node = find_node(frame.label, id=frame.id)

    if frame_node:
        if fail_on_exists:
            raise EntryDoesExistExeption("Frame already exists!")
        else:
            return frame_node

    ship_node = find_node(LiShip().label, imo=frame.imo)
    if not ship_node:
        raise ValueError(f"No or multiple ship found for imo {frame.imo}")

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

    classlabel_to_node = {
        k: find_node("Classification", imo=frame.imo, visCode=v)
        for k, v in classlabel_to_vis.items()
    }

    with neo4j_transaction() as tx:
        frame_node = py2neo.Node(frame.label, **frame.__dict__) 
        tx.create(frame_node)

        inspaction_relation = py2neo.Relationship(inspection_node, "HAS_FRAME", frame_node)
        tx.create(inspaction_relation)

        for k, v in frame.__dict__.items():
            if k in classlabel_to_vis:
                if v > classification_threshold:
                    classification_relation = py2neo.Relationship(frame_node, "DEPICTS", classlabel_to_node[k])
                    tx.create(classification_relation)

        return frame_node

def add_similarity(frame_node_1, frame_node_2, distance):
    with neo4j_transaction() as tx:
        relation = py2neo.Relationship(frame_node_1, "SIMILAR_TO", frame_node_2, distance=distance)
        tx.create(relation)