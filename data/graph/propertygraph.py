import py2neo
from py2neo.matching import *
from py2neo.data import Node, Relationship

import vismodel
from vismodel.LiShip import LiShip

import inspection
from inspection.drawing_node import DrawingNode
from inspection.finding_node import FindingNode
from inspection.image_node import ImageNode
from data.inspection.LiInspection import InspectionNode
import environment.settings as stngs

class PropertyGraph():
    def __init__(self):
        self.graph = py2neo.Graph(uri = stngs.NEO4J, auth=(stngs.NEO4J_USER, stngs.NEO4J_PW))


    # Delete graph
    def delete_all_neo4j_database(self):
        self.graph.delete_all()           


    # Get ship nodes
    def get_ships(self):
        node_matcher = NodeMatcher(self.graph)
        node_type = LiShip()
        nodes = node_matcher.match(node_type.label).all()

        return nodes


    # Create inspection node
    def create_ship_inspection(self, inspection_object, inspection_id, inspection_date):
        tx = self.graph.begin()

        # Find ship node
        node_matcher = NodeMatcher(self.graph)
        ship_node_type = LiShip()
        ship_node = node_matcher.match(ship_node_type.label, id=inspection_object, imo=inspection_object).first()
        if ship_node == None:
            return "{ Error: Ship node not found! }"

        # Check if inspection node exists
        inspection_node_type = InspectionNode()
        inspection_node = node_matcher.match(inspection_node_type.label, id=inspection_id, imo=inspection_object, date=inspection_date).first()
        if inspection_node != None:
            return "{ Error: Inspection node already exists! }"

        # Add new inspection node
        new_inspection_node = py2neo.Node(inspection_node_type.label, id=inspection_id, imo=inspection_object, date=inspection_date, )
        tx.create(new_inspection_node)
        tx.create(py2neo.Relationship(ship_node, "HAS_INSPECTION", new_inspection_node))

        tx.commit()

        return new_inspection_node


    # Get inspection node (by id)
    def get_inspection(self, inspection_id):
        node_matcher = NodeMatcher(self.graph)
        node_type = InspectionNode()
        node = node_matcher.match(node_type.label, id=inspection_id).first()

        return node


    # Get inspection nodes (by imo and date)
    def get_inspections(self, imo, date):
        node_matcher = NodeMatcher(self.graph)
        node_type = InspectionNode()

        nodes = None
        if date != None:
            nodes = node_matcher.match(node_type.label, imo=imo, date=date).all()
        else:
            nodes = node_matcher.match(node_type.label, imo=imo).all()

        return nodes


    # Create inspection drawing node
    def create_inspection_drawing(  self, 
                                    inspection_id, 
                                    drawing_id, 
                                    drawing_filename, 
                                    drawing_originX, drawing_originY, 
                                    drawing_m_per_pixel_X, drawing_m_per_pixel_Y):
        tx = self.graph.begin()

        # Find inspection node
        node_matcher = NodeMatcher(self.graph)
        inspection_node_type = InspectionNode()
        inspection_node = node_matcher.match(inspection_node_type.label, id=inspection_id).first()
        if inspection_node == None:
            return "{ Error: Inspection node not found! }"

        # Check if drawing node exists
        drawing_node_type = DrawingNode()
        drawing_node = node_matcher.match(drawing_node_type.label, id=drawing_id, filename=drawing_filename, imo=inspection_node["imo"], date=inspection_node["date"]).first()
        if drawing_node != None:
            return "{ Error: Drawing node exists! }"

        # Add new drawing node
        new_drawing_node = py2neo.Node(drawing_node_type.label, 
                                        id=drawing_id, 
                                        filename=drawing_filename, 
                                        imo=inspection_node["imo"], 
                                        date=inspection_node["date"],
                                        originX=drawing_originX, 
                                        originY=drawing_originY, 
                                        m_per_pixel_X=drawing_m_per_pixel_X,
                                        m_per_pixel_Y=drawing_m_per_pixel_Y)
        tx.create(new_drawing_node)
        tx.create(py2neo.Relationship(inspection_node, "HAS_DRAWING", new_drawing_node))

        tx.commit()

        return new_drawing_node


    # Get inspection drawing node (by id)
    def get_inspection_drawing(self, drawing_id):
        node_matcher = NodeMatcher(self.graph)
        node_type = DrawingNode()
        node = node_matcher.match(node_type.label, id=drawing_id).first()

        return node

    # Create inspection image node
    def create_inspection_image(self, inspection_id, asset_id, image_id, 
                                    img_filename, video_id,
                                    classifier_results, 
                                    image_location_x, image_location_y, image_location_z, description):
        tx = self.graph.begin()
        node_matcher = NodeMatcher(self.graph)

        # Find asset node        
        class_node = node_matcher.match('Classification', visCode=asset_id).first()
        if class_node == None:
            return "{ Error: Class node not found! }"

        # Find inspection node
        node_type = InspectionNode()
        inspection_node = node_matcher.match(node_type.label, id=inspection_id).first()
        if inspection_node == None:
            return "{ Error: Inspection node not found! }"

        # Check if image node exists
        image_node_type = ImageNode()
        image_node = node_matcher.match(image_node_type.label, id=image_id).first()

        if image_node == None:
            # Add new image node
            image_node = py2neo.Node(image_node_type.label, id=image_id, imo=inspection_node["imo"], date=inspection_node["date"], visCode=asset_id,
                                        img_filename=img_filename, video_id=video_id,
                                        classifier_results=classifier_results, 
                                        location_x_m=image_location_x, location_y_m=image_location_y, location_z=image_location_z,
                                        description=description)
            tx.create(image_node)

        # Create relationship with image node
        tx.create(py2neo.Relationship(class_node, "HAS_IMAGE", image_node))
        tx.create(py2neo.Relationship(inspection_node, "HAS_IMAGE", image_node))
        tx.commit()

        return image_node

    # Create inspection finding node
    def create_inspection_finding(self, inspection_id, finding_id, finding_visCode, finding_description):
        tx = self.graph.begin()

        # Find inspection node
        node_matcher = NodeMatcher(self.graph)
        inspection_node_type = InspectionNode()
        inspection_node = node_matcher.match(inspection_node_type.label, id=inspection_id).first()
        if inspection_node == None:
            return "{ Error: Inspection node not found! }"

        # Check if finding node exists
        finding_node_type = FindingNode()
        finding_node = node_matcher.match(finding_node_type.label, id=finding_id).first()
        if finding_node != None:
            return "{ Error: Finding node already exists! }"

        # Add new finding node
        new_finding_node = py2neo.Node(finding_node_type.label, 
                                       id=finding_id, visCode=finding_visCode, description=finding_description,
                                       imo=inspection_node["imo"], date=inspection_node["date"])
        tx.create(new_finding_node)
        tx.create(py2neo.Relationship(inspection_node, "HAS_FINDING", new_finding_node))

        # Add new relationships
        viscode_node = node_matcher.match("Classification", imo=inspection_node["imo"], visCode=finding_visCode).first()
        tx.create(py2neo.Relationship(viscode_node, "HAS_FINDING", new_finding_node))
        tx.create(py2neo.Relationship(new_finding_node, "FINDING_FOR", viscode_node))

        tx.commit()

        return new_finding_node


    # Get inspection finding node (by id)
    def get_inspection_finding(self, finding_id):
        node_matcher = NodeMatcher(self.graph)
        node_type = FindingNode()
        node = node_matcher.match(node_type.label, id=finding_id).first()

        return node


    # Get inspection finding nodes (by imo, date and visCode)
    def get_inspection_findings(self, imo, date, visCode):
        node_matcher = NodeMatcher(self.graph)
        node_type = FindingNode()

        nodes = None
        if imo == None and date == None and visCode == None:
            nodes = node_matcher.match(node_type.label).all()
        if imo != None and date == None and visCode == None:
            nodes = node_matcher.match(node_type.label, imo=imo).all()
        if imo != None and date != None and visCode == None:
            nodes = node_matcher.match(node_type.label, imo=imo, date=date).all()
        if imo != None and date != None and visCode != None:
            nodes = node_matcher.match(node_type.label, imo=imo, date=date, visCode=visCode).all()
        if imo != None and date == None and visCode != None:
            nodes = node_matcher.match(node_type.label, imo=imo, visCode=visCode).all()
        if imo == None and date != None and visCode == None:
            nodes = node_matcher.match(node_type.label, date=date).all()
        if imo == None and date != None and visCode != None:
            nodes = node_matcher.match(node_type.label, date=date, visCode=visCode).all()
        if imo == None and date == None and visCode != None:
            nodes = node_matcher.match(node_type.label, visCode=visCode).all()

        return nodes


    # Create inspection finding image node
    def create_inspection_finding_image(self, finding_id, image_id, 
                                        img_filename, video_id,
                                        classifier_results, 
                                        image_location_x, image_location_y, image_location_z, description):
        tx = self.graph.begin()

        # Find finding node
        node_matcher = NodeMatcher(self.graph)
        finding_node_type = FindingNode()
        finding_node = node_matcher.match(finding_node_type.label, id=finding_id).first()
        if finding_node == None:
            return "{ Error: Finding node does not exist! }"

        # Check if image node exists
        image_node_type = ImageNode()
        image_node = node_matcher.match(image_node_type.label, id=image_id).first()

        if image_node == None:
            # Add new image node
            image_node = py2neo.Node(image_node_type.label, id=image_id, imo=finding_node["imo"], date=finding_node["date"], visCode=finding_node["visCode"],
                                        img_filename=img_filename, video_id=video_id,
                                        classifier_results=classifier_results, 
                                        location_x_m=image_location_x, location_y_m=image_location_y, location_z=image_location_z,
                                        description=description)
            tx.create(image_node)

        # Create relationship with image node
        tx.create(py2neo.Relationship(finding_node, "HAS_IMAGE", image_node))
        tx.commit()

        return image_node


    # Get inspection finding image node (by id)
    def get_inspection_finding_image(self, image_id):
        node_matcher = NodeMatcher(self.graph)
        node_type = ImageNode()
        node = node_matcher.match(node_type.label, id=image_id).first()

        return node


    # Get inspection finding image nodes (by imo, date and visCode)
    def get_inspection_finding_images(self, imo, date, visCode):
        node_matcher = NodeMatcher(self.graph)
        node_type = ImageNode()

        nodes = None
        if imo == None and date == None and visCode == None:
            nodes = node_matcher.match(node_type.label).all()
        if imo != None and date == None and visCode == None:
            nodes = node_matcher.match(node_type.label, imo=imo).all()
        if imo != None and date != None and visCode == None:
            nodes = node_matcher.match(node_type.label, imo=imo, date=date).all()
        if imo != None and date != None and visCode != None:
            nodes = node_matcher.match(node_type.label, imo=imo, date=date, visCode=visCode).all()
        if imo != None and date == None and visCode != None:
            nodes = node_matcher.match(node_type.label, imo=imo, visCode=visCode).all()
        if imo == None and date != None and visCode == None:
            nodes = node_matcher.match(node_type.label, date=date).all()
        if imo == None and date != None and visCode != None:
            nodes = node_matcher.match(node_type.label, date=date, visCode=visCode).all()
        if imo == None and date == None and visCode != None:
            nodes = node_matcher.match(node_type.label, visCode=visCode).all()

        return nodes
