from data.datastore import EntryDoesExistExeption, find_node, liaci_graph, neo4j_transaction
from data.vismodel.LiShip import LiShip
from py2neo.matching import NodeMatcher


import py2neo



def create(ship: LiShip, fail_on_exists = False):
    ship_node = find_node(ship.label, imo=ship.imo, id=ship.id)
    if ship_node:
        if fail_on_exists:
            raise EntryDoesExistExeption("Ship already exists!")
        else:
            return ship_node
    with neo4j_transaction() as tx:
        node_ship = py2neo.Node(ship.label, id=ship.id, imo=ship.imo, name=ship.name, type=ship.type, marine_traffic_url=ship.marine_traffic_url) 
        tx.create(node_ship)

        # Propeller Arrangement nodes
        obj = ship.LiPropellerArrangement
        node_01 = py2neo.Node(obj.name, "Classification", imo=obj.imo, name=obj.name, visCode=obj.visCode, inspectionCheckPoints=obj.inspectionCheckPoints, findings=obj.findings)
        tx.create(node_01)   
        tx.create(py2neo.Relationship(node_ship, "HAS", node_01))
        
        obj = ship.LiPropellerBladeSealingTightness
        node_02 = py2neo.Node(obj.name, "Classification", imo=obj.imo, name=obj.name, visCode=obj.visCode, inspectionCheckPoints=obj.inspectionCheckPoints, findings=obj.findings)
        tx.create(node_02)   
        tx.create(py2neo.Relationship(node_01, "HAS", node_02))

        # Ship Hull Structure node
        obj = ship.LiShipHullStructure
        node_01 = py2neo.Node(obj.name, "Classification", imo=obj.imo, name=obj.name, visCode=obj.visCode, inspectionCheckPoints=obj.inspectionCheckPoints, findings=obj.findings)
        tx.create(node_01)   
        tx.create(py2neo.Relationship(node_ship, "HAS", node_01))

        # Sea Water System nodes
        obj = ship.LiSeaWaterSystem
        node_01 = py2neo.Node(obj.name, "Classification", imo=obj.imo, name=obj.name, visCode=obj.visCode, inspectionCheckPoints=obj.inspectionCheckPoints, findings=obj.findings)
        tx.create(node_01)   
        tx.create(py2neo.Relationship(node_ship, "HAS", node_01))

        obj = ship.LiOpenings
        node_02 = py2neo.Node(obj.name, "Classification", imo=obj.imo, name=obj.name, visCode=obj.visCode, inspectionCheckPoints=obj.inspectionCheckPoints, findings=obj.findings)
        tx.create(node_02)   
        tx.create(py2neo.Relationship(node_01, "HAS", node_02))

        # Fresh Water System nodes
        obj = ship.LiFreshWaterSystem
        node_01 = py2neo.Node(obj.name, "Classification", imo=obj.imo, name=obj.name, visCode=obj.visCode, inspectionCheckPoints=obj.inspectionCheckPoints, findings=obj.findings)
        tx.create(node_01)   
        tx.create(py2neo.Relationship(node_ship, "HAS", node_01))

        obj = ship.LiBoxCooler
        node_02 = py2neo.Node(obj.name, "Classification", imo=obj.imo, name=obj.name, visCode=obj.visCode, inspectionCheckPoints=obj.inspectionCheckPoints, findings=obj.findings)
        tx.create(node_02)   
        tx.create(py2neo.Relationship(node_01, "HAS", node_02))

        # Motion and Trim Control Arrangement nodes
        obj = ship.LiMotionTrimControlArrangement
        node_01 = py2neo.Node(obj.name, "Classification", imo=obj.imo, name=obj.name, visCode=obj.visCode, inspectionCheckPoints=obj.inspectionCheckPoints, findings=obj.findings)
        tx.create(node_01)   
        tx.create(py2neo.Relationship(node_ship, "HAS", node_01))

        obj = ship.LiStabilisingFins
        node_02 = py2neo.Node(obj.name, "Classification", imo=obj.imo, name=obj.name, visCode=obj.visCode, inspectionCheckPoints=obj.inspectionCheckPoints, findings=obj.findings)
        tx.create(node_02)   
        tx.create(py2neo.Relationship(node_01, "HAS", node_02))

        obj = ship.LiBilgeKeels
        node_03 = py2neo.Node(obj.name, "Classification", imo=obj.imo, name=obj.name, visCode=obj.visCode, inspectionCheckPoints=obj.inspectionCheckPoints, findings=obj.findings)
        tx.create(node_03)   
        tx.create(py2neo.Relationship(node_01, "HAS", node_03))

        # Main Structure nodes
        obj = ship.LiMainStructure
        node_01 = py2neo.Node(obj.name, "Classification", imo=obj.imo, name=obj.name, visCode=obj.visCode, inspectionCheckPoints=obj.inspectionCheckPoints, findings=obj.findings)
        tx.create(node_01)   
        tx.create(py2neo.Relationship(node_ship, "HAS", node_01))

        obj = ship.LiCoating
        node_02 = py2neo.Node(obj.name, "Classification", imo=obj.imo, name=obj.name, visCode=obj.visCode, inspectionCheckPoints=obj.inspectionCheckPoints, findings=obj.findings)
        tx.create(node_02)   
        tx.create(py2neo.Relationship(node_01, "HAS", node_02))

        obj = ship.LiAnodes
        node_03 = py2neo.Node(obj.name, "Classification", imo=obj.imo, name=obj.name, visCode=obj.visCode, inspectionCheckPoints=obj.inspectionCheckPoints, findings=obj.findings)
        tx.create(node_03)   
        tx.create(py2neo.Relationship(node_01, "HAS", node_03))

        # Rudder Arrangement nodes
        obj = ship.LiRudderArrangement
        node_01 = py2neo.Node(obj.name, "Classification", imo=obj.imo, name=obj.name, visCode=obj.visCode, inspectionCheckPoints=obj.inspectionCheckPoints, findings=obj.findings)
        tx.create(node_01)   
        tx.create(py2neo.Relationship(node_ship, "HAS", node_01))

        obj = ship.LiRudderStock
        node_02 = py2neo.Node(obj.name, "Classification", imo=obj.imo, name=obj.name, visCode=obj.visCode, inspectionCheckPoints=obj.inspectionCheckPoints, findings=obj.findings)
        tx.create(node_02)   
        tx.create(py2neo.Relationship(node_01, "HAS", node_02))

        obj = ship.LiRudder
        node_03 = py2neo.Node(obj.name, "Classification", imo=obj.imo, name=obj.name, visCode=obj.visCode, inspectionCheckPoints=obj.inspectionCheckPoints, findings=obj.findings)
        tx.create(node_03)   
        tx.create(py2neo.Relationship(node_01, "HAS", node_03))

        obj = ship.LiSolePiecePintles
        node_04 = py2neo.Node(obj.name, "Classification", imo=obj.imo, name=obj.name, visCode=obj.visCode, inspectionCheckPoints=obj.inspectionCheckPoints, findings=obj.findings)
        tx.create(node_04)   
        tx.create(py2neo.Relationship(node_01, "HAS", node_04))

        obj = ship.LiFlapBeckerRudder
        node_05 = py2neo.Node(obj.name, "Classification", imo=obj.imo, name=obj.name, visCode=obj.visCode, inspectionCheckPoints=obj.inspectionCheckPoints, findings=obj.findings)
        tx.create(node_05)   
        tx.create(py2neo.Relationship(node_01, "HAS", node_05))

        # Propeller Shaft Arrangement nodes
        obj = ship.LiPropellerShaftArrangement
        node_01 = py2neo.Node(obj.name, "Classification", imo=obj.imo, name=obj.name, visCode=obj.visCode, inspectionCheckPoints=obj.inspectionCheckPoints, findings=obj.findings)
        tx.create(node_01)   
        tx.create(py2neo.Relationship(node_ship, "HAS", node_01))

        obj = ship.LiShaftSealTightness
        node_02 = py2neo.Node(obj.name, "Classification", imo=obj.imo, name=obj.name, visCode=obj.visCode, inspectionCheckPoints=obj.inspectionCheckPoints, findings=obj.findings)
        tx.create(node_02)   
        tx.create(py2neo.Relationship(node_01, "HAS", node_02))

        obj = ship.LiShaftPropellerKeyArrangement
        node_03 = py2neo.Node(obj.name, "Classification", imo=obj.imo, name=obj.name, visCode=obj.visCode, inspectionCheckPoints=obj.inspectionCheckPoints, findings=obj.findings)
        tx.create(node_03)   
        tx.create(py2neo.Relationship(node_01, "HAS", node_03))   

        # Propulsion Thruster Arrangement nodes
        obj = ship.LiPropulsionThrusterArrangement
        node_01 = py2neo.Node(obj.name, "Classification", imo=obj.imo, name=obj.name, visCode=obj.visCode, inspectionCheckPoints=obj.inspectionCheckPoints, findings=obj.findings)
        tx.create(node_01)   
        tx.create(py2neo.Relationship(node_ship, "HAS", node_01))

        obj = ship.LiHydraulicOilTightness
        node_02 = py2neo.Node(obj.name, "Classification", imo=obj.imo, name=obj.name, visCode=obj.visCode, inspectionCheckPoints=obj.inspectionCheckPoints, findings=obj.findings)
        tx.create(node_02)   
        tx.create(py2neo.Relationship(node_01, "HAS", node_02))

        # Maneuvering Thruster Arrangement nodes
        obj = ship.LiManeuveringThrusterArrangement
        node_01 = py2neo.Node(obj.name, "Classification", imo=obj.imo, name=obj.name, visCode=obj.visCode, inspectionCheckPoints=obj.inspectionCheckPoints, findings=obj.findings)
        tx.create(node_01)   
        tx.create(py2neo.Relationship(node_ship, "HAS", node_01))

        # All other underwater appendages nodes
        obj = ship.LiAllUnderwaterAppendages
        node_01 = py2neo.Node(obj.name, "Classification", imo=obj.imo, name=obj.name, visCode=obj.visCode, inspectionCheckPoints=obj.inspectionCheckPoints, findings=obj.findings)
        tx.create(node_01)   
        tx.create(py2neo.Relationship(node_ship, "HAS", node_01))

        return node_ship
