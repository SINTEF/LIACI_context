from data.inspection.LiInspection import LiInspection
from py2neo import Node, Relationship

from data.datastore import EntryDoesExistExeption, find_node, neo4j_transaction

def create(inspection: LiInspection, fail_on_exists = False) -> Node:
    inspection_node = find_node(inspection.label, id=inspection.id)
    if inspection_node:
        if fail_on_exists:
            raise EntryDoesExistExeption("Inspection exists already.")
        else:
            return inspection_node

    ship_node = find_node("Ship", imo=inspection.imo)
    if not ship_node:
        raise ValueError(f"Ship with imo {inspection.imo} not found")

    inspection_node = Node(inspection.label, id=inspection.id, imo=inspection.imo, date=inspection.date)
    inspection_relation = Relationship(ship_node, "HAS_INSPECTION", inspection_node)

    with neo4j_transaction() as tx:
        tx.create(inspection_node)
        tx.create(inspection_relation)

    return inspection_node