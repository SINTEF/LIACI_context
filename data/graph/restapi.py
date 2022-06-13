from typing import Optional
from fastapi import FastAPI, Response, HTTPException, status
from fastapi.encoders import jsonable_encoder

import vismodel
from vismodel.LiShip import LiShip
from LiModelFindings import Findings
from LiModelInspectionCheckList import InspectionCheckList
from LiModelInspectionImages import InspectionImages

import json
from datetime import date

import graph
from graph.properties import DrawingProperties, FindingProperties, ImageProperties, InspectionProperties, ShipProperties, VideoProperties
from graph.propertygraph import PropertyGraph

import localstore
from localstore.importer import LocalStoreImporter

DEBUG = 0
if DEBUG:
    import uvicorn

    app = FastAPI()

tags_metadata = [
    {
        "name": "import",
        "description": "Manage imports"
    },
    {
        "name": "ships",
        "description": "Manage ships"
    },
    {
        "name": "inspections",
        "description": "Manage inspections"
    },
    {
        "name": "drawings",
        "description": "Manage drawings"
    },
    {
        "name": "findings",
        "description": "Manage findings"
    },
    {
        "name": "images",
        "description": "Manage images"
    },
]

app = FastAPI(
    title="LIACi Knowledge Graph API",
    description="REST API for the Knowledge Graph in the LIACi Contextualization Module",
    version="0.0.2",
    openapi_tags=tags_metadata)


@app.get("/api/ping", tags=["ping"])
def ping():
    return {"Hello World"}


# Import inspection (id)
@app.get("/api/import/inspection/{id}", tags=["import"])
def import_inspection_data(id: str):
    result = LocalStoreImporter().import_inspection_data(id)
    return Response(content=result, media_type="application/json")


# Create ship (IMO)
@app.post("/api/ships", tags=["ships"])
async def create_ship(ship: ShipProperties):
    myShip = LiShip(ship.id, ship.imo, ship.name, ship.type, ship.marine_traffic_url)
    result = LiShip(ship.id).read()
    if result == None:
        result = myShip.write()
    else:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT , detail="Ship already exists!")
    return Response(content=json.dumps(result), media_type="application/json")


# Get ship
@app.get("/api/ships/{id}", tags=["ships"])
async def get_ship(id: str):
    result = LiShip(id).read()
    if result == None:
        raise HTTPException(status_code=204, detail="Item not found!")
    return Response(content=json.dumps(result), media_type="application/json")


# Get ships
@app.get("/api/ships", tags=["ships"])
async def get_ship():
    result = PropertyGraph().get_ships()
    return Response(content=json.dumps(result), media_type="application/json")

# Create ship inspection
@app.post("/api/inspections", tags=["inspections"])
async def create_inspection(inspection: InspectionProperties):
    result = PropertyGraph().get_inspection(inspection.id)
    if result == None:
        result = PropertyGraph().create_ship_inspection(inspection.inspection_object, inspection.id, jsonable_encoder(inspection.date))
    else:
        raise HTTPException(status_code=409 , detail="Inspection already exists!")
    return Response(content=json.dumps(result), media_type="application/json")


# Get inspection (by id)
@app.get("/api/inspections/{id}", tags=["inspections"])
async def get_inspection(id: str):
    result = PropertyGraph().get_inspection(id)
    return Response(content=json.dumps(result), media_type="application/json")


# Get inspections (by imo and date)
@app.get("/api/inspections/", tags=["inspections"])
async def get_inspections(imo: str, date: Optional[date]=None):
    result = PropertyGraph().get_inspections(imo, jsonable_encoder(date))
    return Response(content=json.dumps(result), media_type="application/json")


# Create inspection drawing
@app.post("/api/inspections/{id}/drawings/", tags=["drawings"])
async def create_inspection_drawing(id: str, drawing: DrawingProperties):
    result = PropertyGraph().get_inspection_drawing(id)
    if result == None:
        result = PropertyGraph().create_inspection_drawing(id, 
                                                        drawing.id, 
                                                        drawing.filename, 
                                                        drawing.originX, 
                                                        drawing.originY, 
                                                        drawing.m_per_pixel_X,
                                                        drawing.m_per_pixel_Y)
    else:
        raise HTTPException(status_code=409 , detail="Inspection drawing already exists!")

    return Response(content=json.dumps(result), media_type="application/json")


# Get inspection drawing (by id)
@app.get("/api/inspections/drawings/{id}", tags=["drawings"])
async def get_inspection_drawing(id: str):
    result = PropertyGraph().get_inspection_drawing(id)
    return Response(content=json.dumps(result), media_type="application/json")


# Create inspection finding
@app.post("/api/inspections/{id}/findings", tags=["findings"])
async def create_inspection_finding(id: str, finding: FindingProperties):
    result = PropertyGraph().get_inspection_finding(id)
    if result == None:
        result = PropertyGraph().create_inspection_finding(id, finding.id, finding.visCode, finding.description)
    else:
        raise HTTPException(status_code=409 , detail="Finding class already exists!")
    
    return Response(content=json.dumps(result), media_type="application/json")


# Get inspection finding (by id)
@app.get("/api/inspections/findings/{id}", tags=["findings"])
async def get_inspection_finding(id: str):
    result = PropertyGraph().get_inspection_finding(id)
    return Response(content=json.dumps(result), media_type="application/json")

# Get inspection findings (by imo, date and visCode)
@app.get("/api/inspections/findings/", tags=["findings"])
async def get_inspection_findings(imo: Optional[str]=None, date: Optional[date]=None, visCode: Optional[str]=None):
    result = PropertyGraph().get_inspection_findings(imo, jsonable_encoder(date), visCode)
    return result


# Create inspection finding image
@app.post("/api/inspections/findings/{id}/images", tags=["images"])
async def create_inspection_finding_image(id: str, image: ImageProperties):
    result = PropertyGraph().create_inspection_finding_image(id, 
                                                            image.id, image.img_filename, image.video_id,
                                                            image.classifier_results, 
                                                            image.location_x_m, image.location_y_m, image.location_z_m,
                                                            image.description)

    return Response(content=json.dumps(result), media_type="application/json")

# Create asset class image
@app.post("/api/inspections/{id}/{asset_id}/images", tags=["images"])
async def create_inspection_image(id: str, asset_id: str, image: ImageProperties):
    result = PropertyGraph().create_inspection_image(id, asset_id,
                                                        image.id, image.img_filename, image.video_id,
                                                        image.classifier_results, 
                                                        image.location_x_m, image.location_y_m, image.location_z_m,
                                                        image.description)

    return Response(content=json.dumps(result), media_type="application/json") 

# Get inspection finding image (by id)
@app.get("/api/inspections/findings/images/{id}", tags=["images"])
async def get_inspection_finding_image(id: str):
    result = PropertyGraph().get_inspection_finding_image(id)
    return Response(content=json.dumps(result), media_type="application/json")


# Get inspection finding images (by imo, date and visCode)
@app.get("/api/inspections/findings/images/", tags=["images"])
async def get_inspection_finding_images(imo: Optional[str]=None, date: Optional[date]=None, visCode: Optional[str]=None):
    result = PropertyGraph().get_inspection_finding_images(imo, jsonable_encoder(date), visCode)
    return Response(content=json.dumps(result), media_type="application/json")


@app.get("/component/{imo}/{visCode}")
def get_component(imo: str, visCode: str):
    obj = LiShip(imo).read(visCode)
    return Response(content=json.dumps(obj), media_type="application/json")


@app.get("/viscodes/all/{imo}")
async def get_all_viscodes(imo: str):
    obj = LiShip(imo).get_vis_codes()
    return Response(content=json.dumps(obj), media_type="application/json")

if DEBUG:
    if __name__ == "__main__":
        uvicorn.run(app, host="0.0.0.0", port=8000)
