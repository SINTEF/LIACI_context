# LIACi data contextualization
Code in this repository is created within the [LIACi project ](https://www.sintef.no/en/projects/2021/liaci/) by SINTEF Digital.
It is published in the paper "Fusion of Multi-Modal Data from Underwater Ship Inspection with Knowledge Graphs" and is also content of the master thesis by Joseph Hirsch with the same name.
## Overview
The LIACi project aims at improving decision support systems in underwater ship inspections.
One big goal is to make use of the onging development in computer vision.
The inference results of machine learning models are contextualized with the available data from the inspection vehicles.
This enables a semantic browsability of the data and also big data analytics for example in a business intelligence context of a whole fleet of ships.
Semantic similarities and clusters complete this feature set.

# Contents of this repository
This repository contains a data contextualization pipeline for underwater ship inspection videos where also telemetry data is available in .ass format and a demonstrator web application that conatains different visualizations for this data.

Please refer to the readmes of the respective directories for further information about the different elements.
## Directory structure
|Directory|Content|
|-|-|
[/data](data)|- domain model for underwater ship inspections<br>- data access scripts for objects from the domain model<br>- data store that handles connections to the neo4j database<br>- Place for thumbnails of video frames and mosaics accessible from both the demonstrator as well as the pipeline
[/demonstrator](demonstrator)|Web application containing different kinds of visulizations on the data of the knowledge graph
[/pipeline](pipeline)|- Inference pipeline for data enrichment with computer vision models<br>- Contextualization pipeline to find similarities and semantic clusters in the inspection video frames

## Architecture
The architecture follows a four layer achritecture as suggested for knowledge graph data contextualization applications [in this paper](https://ieeexplore.ieee.org/document/9779654) by Waszak et al.
It consists of the following layers

|Layer|task(s)|technology|where in this repo|
|-|-|-|-|
|**Presentation**|Presenting the data to the user such that additional knowledge and value is created|![plotly dash framework](https://raw.githubusercontent.com/plotly/dash/dev/components/dash-table/tests/selenium/assets/logo.png)|[/demonstrator](./demonstrator/)|
|**Application**|API for data storage and retrieval|<img width=50px, src="https://upload.wikimedia.org/wikipedia/commons/c/c3/Python-logo-notext.svg" alt="python"/> <img src="https://py2neo.org/2021.1/_static/py2neo-2018.291x50.png" width=100px alt="py2neo"/>|[/data/access](./data/access/)|
|**Business**|Business logic of the inference pipeline, machine learning models, computer vision, data contextualization|<img width=50px, src="https://upload.wikimedia.org/wikipedia/commons/c/c3/Python-logo-notext.svg" alt="python"/> <img height=50px, src="https://raw.githubusercontent.com/microsoft/onnxruntime/main/docs/images/ONNX_Runtime_logo_dark.png" alt="onnx"/> <img src="https://opencv.org/wp-content/uploads/2020/07/OpenCV_logo_black_.png" height=50px alt="cv2"/>|[/pipeline](./pipeline/)|
|**Data**|Data persistande, domain model|![neo4j graph database](https://dist.neo4j.com/wp-content/uploads/20210423072428/neo4j-logo-2020-1.svg)|[/data/vismodel](./data/vismodel/), [/data/inspection](./data/inspection/)<br> and the neo4j database|