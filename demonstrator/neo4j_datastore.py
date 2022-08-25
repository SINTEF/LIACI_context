from os import environ
import py2neo
from py2neo import Node, Relationship
from py2neo.matching import NodeMatcher
from py2neo.matching import RelationshipMatcher

class EntryDoesExistExeption(Exception):
    pass


def find_node(*args, **kwargs):
    node_matcher = NodeMatcher(liaci_graph())
    results = node_matcher.match(*args, **kwargs)
    if results.count() != 1:
        return None
    return results.first()

def find_relation(*args, **kwargs):
    rel_matcher = RelationshipMatcher(liaci_graph())
    results = rel_matcher.match(*args, **kwargs)
    if results.count() != 1:
        return None
    return results.first()

def find_classification_node(visCode, imo):
    query = f"MATCH (c{{visCode: '{visCode}'}}) <-[HAS*]- (s:Ship{{imo: '{imo}'}}) RETURN c"
    with neo4j_transaction() as tx:
        return tx.graph.evaluate(query)
        results = tx.run(query)
        results = [i['c'] for i in results]
        if len(results) == 1:
            py2neoNode = find_node(_id = results[0].identity)
            return py2neoNode
    return None

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


liaci_datastore_graph = None
def liaci_graph(username=None, password=None, host=None, port="7687"):
    if username is None:
        username = environ.get("NEO4J_USERNAME", "neo4j")
    if password is None:
        password = environ.get("NEO4J_PASSWORD", "liaci")
    if host is None:
        host = environ.get("NEO4J_HOST", "localhost")
    if port is None:
        port = environ.get("NEO4J_PORT", "7687")
    global liaci_datastore_graph
    if liaci_datastore_graph is None:
        liaci_datastore_graph = py2neo.Graph(host=host, user=username, port=port, password=password) 
    return liaci_datastore_graph

def delete_all_neo4j_database():
    liaci_graph().delete_all()           
