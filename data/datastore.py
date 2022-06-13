import py2neo
from py2neo import Node, Relationship
from py2neo.matching import NodeMatcher

def find_node(*args, **kwargs):
    node_matcher = NodeMatcher(liaci_graph())
    results = node_matcher.match(*args, **kwargs)
    if results.count() != 1:
        return None
    return results.first()

class neo4j_transaction(object):
    def __init__(self) -> None:
        self.graph = liaci_graph()
    
    def __enter__(self):
        self.transaction = self.graph.begin()
        return self.transaction

    def __exit__(self, type, value, traceback):
        if traceback is None:
            self.transaction.commit()
        else:
            self.transaction.rollback()

def liaci_graph(username="neo4j", password="liaci", host="localhost", port="7687"):
    return py2neo.Graph(host=host, user=username, port=port, password=password) 

def delete_all_neo4j_database():
    liaci_graph().delete_all()           
