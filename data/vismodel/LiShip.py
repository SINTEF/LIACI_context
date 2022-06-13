from data.vismodel.LiShipHullStructure import *
from data.vismodel.LiSeaWaterSystem import *
from data.vismodel.LiFreshWaterSystem import *
from data.vismodel.LiMotionTrimControlArrangement import *
from data.vismodel.LiMainStructure import *
from data.vismodel.LiPropellerArrangement import *
from data.vismodel.LiPropellerShaftArrangement import *
from data.vismodel.LiRudderArrangement import *
from data.vismodel.LiPropulsionThrusterArrangement import *
from data.vismodel.LiManeuveringThrusterArrangement import *
from data.vismodel.LiAllUnderwaterAppendages import *


from data.inspection.drawing_node import DrawingNode


class LiShip():
    def __init__(self, id=None, imo=None, name=None, type=None, marine_traffic_url=None):
        self.label = "Ship"
        self.id = id
        self.imo = imo
        self.name = name
        self.type = type
        self.marine_traffic_url = marine_traffic_url
        self.LiShipHullStructure = LiShipHullStructure(imo)
        self.LiPropellerArrangement = LiPropellerArrangement(imo)
        self.LiPropellerBladeSealingTightness = LiPropellerBladeSealingTightness(imo)
        self.LiPropellerShaftArrangement = LiPropellerShaftArrangement(imo)
        self.LiShaftSealTightness = LiShaftSealTightness(imo)
        self.LiShaftPropellerKeyArrangement = LiShaftPropellerKeyArrangement(imo)
        self.LiSeaWaterSystem = LiSeaWaterSystem(imo)
        self.LiOpenings = LiOpenings(imo)
        self.LiFreshWaterSystem = LiFreshWaterSystem(imo)
        self.LiBoxCooler = LiBoxCooler(imo)
        self.LiMotionTrimControlArrangement = LiMotionTrimControlArrangement(imo)
        self.LiStabilisingFins = LiStabilisingFins(imo)
        self.LiBilgeKeels = LiBilgeKeels(imo)
        self.LiMainStructure = LiMainStructure(imo)
        self.LiCoating = LiCoating(imo)
        self.LiAnodes = LiAnodes(imo)
        self.LiRudderArrangement = LiRudderArrangement(imo)
        self.LiRudderStock = LiRudderStock(imo)
        self.LiRudder = LiRudder(imo)
        self.LiSolePiecePintles = LiSolePiecePintles(imo)
        self.LiFlapBeckerRudder = LiFlapBeckerRudder(imo)
        self.LiPropulsionThrusterArrangement = LiPropulsionThrusterArrangement(imo)
        self.LiHydraulicOilTightness = LiHydraulicOilTightness(imo)
        self.LiManeuveringThrusterArrangement = LiManeuveringThrusterArrangement(imo)
        self.LiAllUnderwaterAppendages = LiAllUnderwaterAppendages(imo)
        self.Inspections = list()
        self.Drawings = list()
        self.Owner=None
    

    def add_inspection(self, inspection=None):
        if inspection==None:
            inspection = LiInspection.LiInspection(customer_name=self.Owner, ship=self)
        self.Inspections.append(inspection)


    def add_drawing(self, drawing=None):
        if drawing==None:
            drawing = LiDrawing.LiDrawing(customer_name=self.Owner, ship=self)
        self.Drawings.append(drawing)

    def read(self, viscode=None):
        node_matcher = NodeMatcher(self.graph)

        return node_matcher.match(id=self.id).first()


    def add_findings(self, date, visCode):
        graph = py2neo.Graph(uri = stngs.NEO4J, auth=(stngs.NEO4J_USER, stngs.NEO4J_PW))

        tx = graph.begin()

        nodes = NodeMatcher(graph)
        ship_node = nodes.match("Ship", imo=self.imo).first()
        #inspection_nodes = nodes.match("Inspection", date=date)

        #rel = graph.relationships.match((ship_node, inspection_nodes), "HAS_INSPECTION")
        
        #inspection_node = ship_node.traverse("HAS_INSPECTION").where("date", "=", date)
        #query = "MATCH (s:LiShip)-[r:HAS_INSPECTION]->(i:LiInspection) WHERE i.date = date"

        inspection_node = None
        relmatcher = RelationshipMatcher(graph)

        end_node = None
        size = len(graph.match((ship_node, end_node), "HAS_INSPECTION"))
        print("********* size = ", str(size))

#        for rel in graph.match(start_node=ship_node, rel_type="HAS_INSPECTION"):
#            if rel.end_node().date == date:
#                inspection_node = rel.end_node()

#        for rel in graph.match((ship_node, end_node), "HAS_INSPECTION"):
#            if rel.end_node().date == date:
#                inspection_node = rel.end_node()
        
        inspection_node = nodes.match("Inspection", imo=self.imo, date=date).first()

#        node_type = LiModelFindings2.Findings2()
#        new_findings_node = py2neo.Node("Findings2", IMO=self.IMO, date=date)
        node_type = LiModelFindings2.Findings2()
        new_findings_node = py2neo.Node("Findings", imo=self.imo, date=date, visCode=visCode, name=node_type.name)
        tx.create(new_findings_node)
        tx.create(py2neo.Relationship(inspection_node, "HAS_FINDINGS", new_findings_node))

        viscode_node = nodes.match("Classification", imo=self.imo, visCode=visCode).first()
        tx.create(py2neo.Relationship(viscode_node, "HAS_FINDINGS", new_findings_node))
        tx.create(py2neo.Relationship(new_findings_node, "FINDINGS_FOR", viscode_node))

        tx.commit()

        return True


    def get_findings(self, date, visCode):
        graph = py2neo.Graph(uri = stngs.NEO4J, auth=(stngs.NEO4J_USER, stngs.NEO4J_PW))

        node_matcher = NodeMatcher(graph)
        node = node_matcher.match("Findings", imo=self.imo, date=date, visCode=visCode).first()

        return node


    def write_findings(self, findings, visCode=None):
        graph = py2neo.Graph(uri = stngs.NEO4J, auth=(stngs.NEO4J_USER, stngs.NEO4J_PW))
        nodes = NodeMatcher(graph)
        node = nodes.match(imo=self.imo, visCode=visCode).first()
        node['findings'] = findings
        graph.push(node)

        return True


    def write_inspection_checklist(self, checklist, visCode=None):
        graph = py2neo.Graph(uri = stngs.NEO4J, auth=(stngs.NEO4J_USER, stngs.NEO4J_PW))
        nodes = NodeMatcher(graph)
        node = nodes.match(imo=self.imo, visCode=visCode).first()
        node['inspectionCheckPoints'] = checklist
        graph.push(node)


    def set_ship_images(self, images, visCode=None):
        graph = py2neo.Graph(uri = stngs.NEO4J, auth=(stngs.NEO4J_USER, stngs.NEO4J_PW))
        nodes = NodeMatcher(graph)
        node = nodes.match(imo=self.imo, visCode=visCode).first()
        if node['inspectionImages'] == None:
            node['inspectionImages'] = images
        else:
            node['inspectionImages'].append(images[0])    
        graph.push(node)

        return True


    def get_vis_codes(self):
        graph = py2neo.Graph(uri = stngs.NEO4J, auth=(stngs.NEO4J_USER, stngs.NEO4J_PW))
        list_all_vis_codes = []
        query = "MATCH (n) WHERE n.imo=" + f'"{self.id}"' + " AND NOT n.visCode='' RETURN n.name, n.visCode"
        nodes = graph.run(query)
        for n in nodes:
            dict_node = {'name': n[0], 'visCode': n[1]}
            list_all_vis_codes.append(dict_node)

        return list_all_vis_codes
