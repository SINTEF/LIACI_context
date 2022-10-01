import pickle
from neo4j_datastore import neo4j_transaction
from dataclasses import dataclass
from dash import Input, Output

@dataclass
class FilterOptions():
    inspections: list
    defects: list
    parts: list
    visual_similarities: float
    telemetry_similarities: float
    image_quality_threshold: float
    mosaics: bool
    clusters: bool
    key_frames: bool
    filter_id: str

def get_findings_where_clause(filter_options, begin, ignore_quality = False):
    findings_where_clause = f'{begin} i.uciqe >= {0 if ignore_quality else filter_options.image_quality_threshold}'
    if len(filter_options.parts) > 0: findings_where_clause += f" AND ( "
    findings_where_clause += ' OR '.join([f"i.{p} > 0.8" for p in filter_options.parts])
    if len(filter_options.parts) > 0: findings_where_clause += " ) "
    
    if len(filter_options.defects) > 0: findings_where_clause += f" AND ( "
    findings_where_clause += ' OR '.join([f"i.{d} > 0.8" for d in filter_options.defects])
    if len(filter_options.defects) > 0: findings_where_clause += " ) "
    return findings_where_clause

def get_inspections():
    query = "MATCH (s:Ship) -[:HAS_INSPECTION]-> (i:Inspection) RETURN DISTINCT s.name+' on '+i.date as i, i.id as id"
    with neo4j_transaction() as tx:
        return {i['i']: i['id'] for i in tx.run(query)}

def get_frames_angle(inspection_id, heading, filter_options = None):
    query = f"""MATCH (insp:Inspection{{id:{inspection_id}}}) -[:HAS_FRAME]-> (i:Image) where (round((i.Heading - coalesce(insp.ship_heading, 0) + 180)/30)*30)%360 = {heading}
        {get_findings_where_clause(filter_options, "AND")}
        {'with i OPTIONAL MATCH (i) -[:IN_MOSAIC]-> (m:Mosaic) with coalesce(m.seg_image_file, i.thumbnail) as image_path, coalesce(m.uciqe, i.uciqe) as uciqe' 
            if filter_options.mosaics else 'with i.thumbnail as image_path, i.uciqe as uciqe'}
        return distinct image_path, uciqe order by uciqe desc
    """
    with neo4j_transaction() as tx:
        cursor = tx.run(query)
        return [{'path': r['image_path'], 'uciqe': r['uciqe']} for r in cursor]

def get_frames_cluster(inspection_id, cluster_id):
    query = f"""MATCH (insp:Inspection{{id:{inspection_id}}}) -[:HAS_FRAME]-> (i:Image{{dbscan_cluster:{cluster_id}}}) return i.thumbnail as image_path, i.uciqe as uciqe order by i.framenumber asc"""
    with neo4j_transaction() as tx:
        cursor = tx.run(query)
        return [{'path': r['image_path'], 'uciqe': r['uciqe']} for r in cursor]


def _q_images_and_inspections(filter_options, ignore_quality = False):
        return f"""MATCH (i:Image) <-[:HAS_FRAME]- (ins:Inspection) WHERE ins.id in [{','.join([f"{i}" for i in filter_options.inspections])}]{get_findings_where_clause(filter_options, " AND", ignore_quality = ignore_quality)}"""

def _q_all_images_and_inspections(filter_options):
        neighbor_images = f"""{_q_images_and_inspections(filter_options)} WITH i, collect(i) as ilist MATCH (i) -[]- (i2:Image)"""
        return f"""{neighbor_images} WITH ilist + collect(i2) as alli UNWIND alli as i MATCH (i:Image) <-[:HAS_FRAME]- (ins:Inspection)"""


def get_graph_stuff(filter_options):

    nodes = {}
    in_inspection = set()
    similar_to = set()
    visually_similar_to = set()
    in_mosaic = set()
    in_cluster = set()
    shows_part = set()
    part_of_ship = set()
    
    LIMIT = 300
    if len(filter_options.inspections) == 0: return {}
    graph_images = _q_images_and_inspections(filter_options)
    query1 = f"""{graph_images} RETURN  i, ins ORDER BY i.id LIMIT {LIMIT}"""
    with neo4j_transaction() as tx:
        cursor = tx.run(query1)
        results = [(r['i'], r['ins']) for r in cursor]
        for image, inspection in results:
            image_id = f"im_{image['id']}"
            inspection_id = f"in_{inspection['id']}"
            nodes[image_id] = dict(image.nodes[0])
            if not inspection_id in nodes: 
                nodes[inspection_id] = dict(inspection.nodes[0])
            in_inspection.add((image_id, inspection_id))


    """
    query2/3: Other Images and their similarity
    """

    query2 = f"""{graph_images} WITH i ORDER BY i.id LIMIT {LIMIT} MATCH (i) -[r:SIMILAR_TO]- (i2:Image) WHERE r.distance < {filter_options.telemetry_similarities} RETURN i.id as i, i2.id as i2"""
    query3 = f"""{graph_images} WITH i ORDER BY i.id LIMIT {LIMIT} MATCH (i) -[r:VISUALLY_SIMILAR_TO]- (i2:Image) WHERE r.distance < {filter_options.visual_similarities} RETURN i.id as i, i2.id as i2"""
    with neo4j_transaction() as tx:
        cursor = tx.run(query2)
        results = [(r['i'], r['i2']) for r in cursor]
        for image1, image2 in results:
            image1_id = f"im_{image1}"
            image2_id = f"im_{image2}"
            similar_to.add((image1_id, image2_id))
            #sim_hist.append(distance)

        cursor = tx.run(query3)
        results = [(r['i'], r['i2']) for r in cursor]
        for image1, image2 in results:
            image1_id = f"im_{image1}"
            image2_id = f"im_{image2}"
            visually_similar_to.add((image1_id, image2_id))
            #vsim_hist.append(distance)



    """
    query4: find mosaics and clusters
    """
    query4 = f"""{graph_images} 
                WITH i ORDER BY i.id LIMIT {LIMIT} MATCH (i) -[:IN_MOSAIC]-> (m:Mosaic)
                WITH m, collect(i.uciqe) as qlist, collect(i) as ilist 
                WITH m,  ilist, reduce(s = 0 , v in qlist | s + v)/size(qlist) as avg 
                UNWIND ilist as i RETURN i, m, avg as quality"""
    if filter_options.mosaics:
        with neo4j_transaction() as tx:
            cursor = tx.run(query4)
            results = [(r['i'], r['m'], r['quality']) for r in cursor]
            for image, mosaic, quality in results:
                image_id = f"im_{image['id']}"
                mosaic_id = f"m_{mosaic['id']}"
                nodes[mosaic_id] = dict(mosaic.nodes[0])
                in_mosaic.add((image_id, mosaic_id))

    if filter_options.clusters:
        query_cluster_nodes = f"""{graph_images} WITH i ORDER BY i.id LIMIT {LIMIT} MATCH (i) -[:IN_CLUSTER]-> (c:Cluster) RETURN i, c"""
        with neo4j_transaction() as tx:
            cursor = tx.run(query_cluster_nodes)
            results = [(r['i'], r['c']) for r in cursor]
            for image, cluster in results:
                image_id = f"im_{image['id']}"
                cluster_id = f"{cluster['id']}"
                if cluster_id.endswith("-1"): continue
                nodes[cluster_id] = dict(cluster.nodes[0])
                in_cluster.add((image_id, cluster_id))


    """
    query5: find parts and ships
    """
    query5 = f"""{graph_images} WITH i ORDER BY i.id LIMIT {LIMIT} MATCH (i) -[:DEPICTS]-> (p) <-[:HAS*]- (s:Ship) RETURN i, p, s"""
    with neo4j_transaction() as tx:
        cursor = tx.run(query5)
        results = [(r['i'], r['p'], r['s']) for r in cursor]
        for image, part, ship in results:
            image_id = f"im_{image['id']}"
            ship_id = f"s_{ship['imo']}"
            part_id = f"p_{ship_id}_{part['visCode']}"

            nodes[ship_id] = dict(ship.nodes[0])
            nodes[part_id] = dict(part.nodes[0])

            shows_part.add((image_id, part_id))
            part_of_ship.add((part_id, ship_id))

# 'nodes': {},
# 'similar_to': [],
# 'visually_similar_to': [],
# 'in_inspection': [],
# 'in_mosaic': [],
# 'in_cluster': [],
# 'shows_part': [],
# 'part_of_ship': [],

    return (nodes, similar_to, visually_similar_to, in_inspection, in_mosaic, in_cluster, shows_part, part_of_ship)

# 'q_hist': [],
# 'd_hist': [],
# 'mg_hist': [],
# 'sim_hist': [],
# 'vsim_hist': [],
def get_histogram_data(filter_options:FilterOptions):

    q_hist = []
    mg_hist = []
    d_hist = []
    sim_hist = []
    vsim_hist = []

    images_and_inspections = _q_images_and_inspections(filter_options, ignore_quality = True)
    q_hist_query = f"""{images_and_inspections} RETURN i.uciqe as uciqe"""
    d_hist_query = f"""{images_and_inspections} RETURN i.Depth as depth"""
    mg_hist_query = f"""{images_and_inspections} WITH i MATCH (i)-[:IN_MOSAIC]-> (m:Mosaic) 
        WITH distinct m.id as id, coalesce(m.marine_growth_percentage,0) as mgp, coalesce(m.ship_hull_percentage,0) as shp            
        WITH CASE WHEN shp < 0.05 THEN 0 ELSE mgp / shp END as mg            
        RETURN CASE WHEN mg>1 THEN 1 ELSE mg END as mg"""
    sim_hist_query = f"""{images_and_inspections} WITH i MATCH (i)-[r:SIMILAR_TO]- () RETURN r.distance as d"""
    vsim_hist_query = f"""{images_and_inspections} WITH i MATCH (i)-[r:VISUALLY_SIMILAR_TO]- () RETURN r.distance as d"""

    with neo4j_transaction() as tx:
        cursor = tx.run(q_hist_query)
        q_hist = [r['uciqe'] for r in cursor]

        cursor = tx.run(d_hist_query)
        d_hist = [r['depth'] for r in cursor]

        cursor = tx.run(mg_hist_query)
        mg_hist = [min(r['mg'] * 100, 100) for r in cursor]

        cursor = tx.run(sim_hist_query)
        sim_hist = [r['d'] for r in cursor]

        cursor = tx.run(vsim_hist_query)
        vsim_hist = [r['d'] for r in cursor]
    
    return q_hist, mg_hist, d_hist, sim_hist, vsim_hist



# 'heading_hist': {},
def get_headings_hist(filter_options):
    heading_hist = {}
    query_headings = f"""match (insp:Inspection) <-[:HAS_INSPECTION]- (s:Ship) where insp.id in [{','.join([f"{i}" for i in filter_options.inspections])}] 
        with s.name as name, insp match (insp) -[:HAS_FRAME]-> (i:Image) {get_findings_where_clause(filter_options, "WHERE")} 
        with name, coalesce(insp.ship_heading, 0) as ship_heading, insp.id as id, insp.date as date, (round((i.Heading - coalesce(insp.ship_heading, 0) + 180)/30)*30)%360 as heading 
        return name, id, date, ship_heading, heading, count(*) as count order by heading asc"""

    with neo4j_transaction() as tx:
        cursor = tx.run(query_headings)
        for r in cursor:
            inspection_id = r['id']
            if not inspection_id in heading_hist:
                heading_hist[inspection_id] = {
                    'ship_name': r['name'],
                    'ship_heading': r['ship_heading'],
                    'date': r['date'],
                    'data': []
                }
            heading_hist[inspection_id]['data'].append({'heading':r['heading'], 'count': r['count']})
    
    return heading_hist

# 'by_ship_table': [],
# 'by_part_table': [],
def get_tables(filter_options):
    images_and_inspections = _q_images_and_inspections(filter_options)
    query6 = f"""{images_and_inspections} WITH i, ins MATCH (ins) <-[:HAS_INSPECTION]- (s:Ship) 
        RETURN s.name as name, s.imo as imo, 
                sum(CASE WHEN i.marine_growth > 0.8 THEN 1 ELSE 0 END) as marine_growth,
                sum(CASE WHEN i.corrosion > 0.8 THEN 1 ELSE 0 END) as corrosion,
                sum(CASE WHEN i.defect > 0.8 THEN 1 ELSE 0 END) as defect,
                sum(CASE WHEN i.paint_peel > 0.8 THEN 1 ELSE 0 END) as paint_peel
        """
    query7 = f"""{images_and_inspections} WITH i MATCH (i) -[:DEPICTS]-> (p) 
        RETURN p.name as name,  
                sum(CASE WHEN i.marine_growth > 0.8 THEN 1 ELSE 0 END) as marine_growth,
                sum(CASE WHEN i.corrosion > 0.8 THEN 1 ELSE 0 END) as corrosion,
                sum(CASE WHEN i.defect > 0.8 THEN 1 ELSE 0 END) as defect,
                sum(CASE WHEN i.paint_peel > 0.8 THEN 1 ELSE 0 END) as paint_peel
        """
    table_ships = []
    table_parts = []
    with neo4j_transaction() as tx:
        cursor = tx.run(query6)
        table_ships = [{key: r[key] for key in 'name imo marine_growth paint_peel corrosion defect'.split()} for r in cursor]

        cursor = tx.run(query7)
        table_parts = [{key: r[key] for key in 'name marine_growth paint_peel corrosion defect'.split()} for r in cursor]

    return table_ships, table_parts


# 'clusters_table': []
def get_cluster_table(filter_options):
    images_and_inspections = _q_images_and_inspections(filter_options)
    query_clusters = f"""{images_and_inspections} WITH ins.id as inspection_id, i 
        MATCH (i) -[:IN_CLUSTER]-> (c:Cluster) WHERE c.number <> -1 WITH inspection_id, c.number as cluster, 
        collect(CASE WHEN i.marine_growth > 0.8 THEN 1 ELSE 0 END) as mg ,
        collect(CASE WHEN i.corrosion > 0.8 THEN 1 ELSE 0 END) as co ,
        collect(CASE WHEN i.paint_peel > 0.8 THEN 1 ELSE 0 END) as pp ,
        collect(CASE WHEN i.defect > 0.8 THEN 1 ELSE 0 END) as de ,
        collect(CASE WHEN i.propeller > 0.8 THEN 1 ELSE 0 END) as pr ,
        collect(CASE WHEN i.over_board_valve > 0.8 THEN 1 ELSE 0 END) as ob ,
        collect(CASE WHEN i.sea_chest_grating > 0.8 THEN 1 ELSE 0 END) as sc ,
        collect(CASE WHEN i.bilge_keel > 0.8 THEN 1 ELSE 0 END) as bk ,
        collect(CASE WHEN i.anode > 0.8 THEN 1 ELSE 0 END) as an ,
        min(i.framenumber) as frame_number,

        collect(i) as nodes


        ORDER BY frame_number

        RETURN cluster, inspection_id, 
            reduce(mg_sum = 0, m in mg | mg_sum + m) as marine_growth ,
            reduce(co_sum = 0, m in co | co_sum + m) as corrosion ,
            reduce(pp_sum = 0, m in pp | pp_sum + m) as paint_peel ,
            reduce(de_sum = 0, m in de | de_sum + m) as defect ,
            reduce(pr_sum = 0, m in pr | pr_sum + m) as propeller ,
            reduce(ob_sum = 0, m in ob | ob_sum + m) as over_board_valve ,
            reduce(sc_sum = 0, m in sc | sc_sum + m) as sea_chest_grating ,
            reduce(bk_sum = 0, m in bk | bk_sum + m) as bilge_keel ,
            reduce(an_sum = 0, m in an | an_sum + m) as anode ,
            size(nodes) as num_nodes,
            reduce(kf = {{image_path:'', uciqe:0}}, i in nodes | CASE WHEN kf.uciqe > i.uciqe THEN kf ELSE {{image_path:i.thumbnail, uciqe:i.uciqe}} END).image_path as key_frame_image_path
            
        order by cluster asc"""
    with neo4j_transaction() as tx:
        cursor = tx.run(query_clusters)
        table_clusters = [
            {'cluster': r['cluster'],
            'inspection_id': r['inspection_id'],
            'id':f'{r["cluster"]}={r["inspection_id"]}',
            'size': r['num_nodes'],
            'key_frame': f"![{r['key_frame_image_path']}](assets/imgs/{'mosaics' if r['key_frame_image_path'].startswith('m') else 'frames'}/{r['key_frame_image_path']})",
            'keywords': ' '.join([kw for kw in 'marine_growth.paint_peel.corrosion.defect.sea_chest_grating.over_board_valve.bilge_keel.propeller.anode'.split('.') if r[kw] > r['num_nodes']//3])} for r in cursor]
        return table_clusters