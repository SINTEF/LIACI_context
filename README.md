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

