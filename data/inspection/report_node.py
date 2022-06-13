import uuid
import os
import subprocess
from subprocess import PIPE
from fpdf import FPDF
import environment.settings as stngs
import requests

class ReportNode(object):
    def __init__(self, ship=None):
        self.name = None
        self.ship = ship
        self.author = None
        self.uuid = uuid.uuid1
        self.IMO = None
        self.VIS = None

    def createPDF(self, file_name = None, IMO = None, VIS = None, classifier_data = None):

        if IMO is not None:

            inspected_component = requests.get(stngs.API_URI + '/component/' + IMO+ '/' + VIS).json()

            pdf = FPDF()
            pdf.add_page()
            pdf.set_font('Arial', 'B', 16)
            pdf.cell(40, 10, 'Work report 1096: MV Island Vanguard')
            pdf.set_xy(10, 25)
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(40, 10, inspected_component['name'])
            pdf.set_font('Arial', '', 12)
            pdf.set_xy(10, 35)
            pdf.multi_cell(180, 5, inspected_component['findings'][0])
            pdf.image('data/' + inspected_component['inspectionImages'][0] + '.jpg', 10, 60, w=120)
            pdf.set_xy(135, 60)
            pdf.multi_cell(60, 5, classifier_data[2]['labels'])
            pdf.set_xy(135, 100)
            pdf.multi_cell(60, 5, 'Image ID: ' + classifier_data[2]['image_uuid'], align='L')
            pdf.set_xy(135, 120)
            pdf.multi_cell(60, 5, 'Snapshot time: ' + classifier_data[2]['snapshot_time'], align='L')
            pdf.image('data/' + inspected_component['inspectionImages'][1] + '.jpg', 10, 135, w=120)
            pdf.set_xy(135, 135)
            pdf.multi_cell(60, 5, classifier_data[1]['labels'])
            pdf.set_xy(135, 175)
            pdf.multi_cell(60, 5, 'Image ID: ' + classifier_data[1]['image_uuid'], align='L')
            pdf.set_xy(135, 195)
            pdf.multi_cell(60, 5, 'Snapshot time: ' + classifier_data[1]['snapshot_time'], align='L')
            pdf.image('data/' + inspected_component['inspectionImages'][2] + '.jpg', 10, 210, w=120)
            pdf.set_xy(135, 210)
            pdf.multi_cell(60, 5, classifier_data[0]['labels'])
            pdf.set_xy(135, 250)
            pdf.multi_cell(60, 5, 'Image ID: ' + classifier_data[0]['image_uuid'], align='L')
            pdf.set_xy(135, 270)
            pdf.multi_cell(60, 5, 'Snapshot time: ' + classifier_data[0]['snapshot_time'], align='L')
            pdf.output('report.pdf', 'F')
    