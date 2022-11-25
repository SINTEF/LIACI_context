# The data module
The data module is providing an API to the neo4j database to use in the infrerence pipeline.
It handles the database connection as well as the creation and update of nodes and relations in its data access scripts [access](./access/).
The module also contains the domain model consisting of data classes in [inspection](./inspection/) and [vismodel](./vismodel/) respectivel respectively.
An overview of the data module can be seen in the following diagram.
![data modules diagram](/doc/data_modules.png)