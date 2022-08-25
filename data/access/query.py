
from typing import List
from data.access.datastore import find_node, liaci_graph, neo4j_transaction
from data.vismodel.LiShip import LiShip
from py2neo.matching import NodeMatcher


import py2neo

def get_labels() -> List[str]:
    result = liaci_graph().query("MATCH (n) RETURN DISTINCT labels(n)")
    labels = []
    for row in result:
        labels.extend(row[0])
    return list(set(labels))
