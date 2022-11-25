from cv2 import threshold
from data.access.datastore import EntryDoesExistExeption, find_classification_node, find_node, find_relation, liaci_graph, neo4j_transaction
from ..inspection.cluster_node import ClusterNode
from data.inspection.image_node import ImageNode, MosaicNode
from data.vismodel.LiShip import LiShip
from data.inspection.LiInspection import LiInspection
from py2neo.matching import NodeMatcher
import py2neo

"""
Searches the database for the cluster node with the given id. If it does not exist,
the cluster node is created. Then the frame node is attached to the cluster with
a IN_CLUSTER relation."""
def create_or_attach(frame: py2neo.Node, cluster_id: str, cluster_number: int):
    cluster_node = find_node(ClusterNode(cluster_id).label, id=cluster_id)
    if cluster_node is None:
        cluster_node = create_cluster(ClusterNode(cluster_id, cluster_number))
    else:
        if find_relation((frame, cluster_node), "IN_CLUSTER") is not None:
            return
    with neo4j_transaction() as tx:
        rel =  py2neo.Relationship(frame, "IN_CLUSTER", cluster_node)
        tx.create(rel)

def create_cluster(cluster_node: ClusterNode):
    n = py2neo.Node(cluster_node.label, **cluster_node.__dict__)
    with neo4j_transaction() as tx:
        tx.create(n)
    return n