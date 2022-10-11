import logging
import os
from typing import List
import click
import sys
from computer_vision.LIACI_stitcher import demo
from video_input.file_input import VideoFileFinder
from video_input.inspection_video_input import InspectionVideoFile
import similarity_pipeline
import scree_plots.scree_plot_pca

import data.access.inspection as inspection_access

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO") )

logger = logging.getLogger('main')

logger.debug("Logger started, Loglevel: at least DEBUG")

@click.group()
def messages():
    """LIACi contextualization tools.\n 


This program contains a set of tools to analyze videos from underwater ship inspections and store the results in a neo4j database.
The neo4j credentials can be stored as envionment variables:\n
NEO4J_HOST for the neo4j host (default: localhost)\n
NEO4J_PORT for the neo4j port (default: 7687)\n
NEO4J_USERNAME for the username (default: neo4j)\n
NEO4J_PASSWORD for the password (default: liaci)\n\n

Please find below a set of commands that can be used with this program:"""
    pass

@click.command()
@click.argument('path')
def list(path):
    """List the available inspections for analysis.

PATH Path to the video files. All subfolders will be searched. Inspections need to have an .mp4 video file and a .ass metadata file with the same name."""

    from video_input.file_input import VideoFileFinder
    vi = VideoFileFinder(path)
    for i, inspection_file in enumerate(vi.list_inspections()):
        print(f"{i:<2}: {inspection_file.video_file: <70}, {inspection_file.video_file_size/1024/1024:12.2f}MB video, {inspection_file.context_file_size/1024/1024:12.2f}MB context")

@click.command()
@click.argument('what', type=click.Choice(['similarities', 'nodes']), )
@click.option('--all', help="Use faster queries for deleting all inspections.", is_flag = True)
def clear(what, all):
    """Delete whole inspections or similarities and clusters of inspections from the neo4j database.

{similarities|nodes} Choose nodes for whole inspections and similarities to only delete clusters and similarities."""
    if 'similarities' == what:
        clear_similarities(all)
    if 'nodes' == what:
        clear_nodes(all)


def clear_nodes(all):
    from data.access.datastore import delete_all_neo4j_database, delete_from_neo4j
    if not all:
        insp = ask_for_stored_inspection_ids()
    else:
        insp = None
    if input("delete from database? (Canot be undone) Y/N ") == "Y":
        if insp is None:
            delete_all_neo4j_database()
        else:
            delete_from_neo4j(insp)
        print("database deleted.")
    else:
        print("database NOT deleted.")

def clear_similarities(all):
    similarity_pipeline.delete_all_similarities(inspection_filter=ask_for_stored_inspection_ids())


def ask_for_selection(vi:VideoFileFinder) -> List[InspectionVideoFile]:
    inspection_files = vi.list_inspections()
    for i, inspection_file in enumerate(inspection_files):
        print(f"{i:<2}: {inspection_file.video_file: <70}, {inspection_file.video_file_size/1024/1024:12.2f}MB video, {inspection_file.context_file_size/1024/1024:12.2f}MB context")

    print()
    if 1 == len(inspection_files):
        selection = 'a'
    else:
        selection = input("Select one or many videos (comma seperated). Type 'a' for all videos.")

    if 'a' == selection:
        pass
    else:
        inspection_files = [inspection for i, inspection in enumerate(inspection_files) if str(i) in selection.replace(" ", "").split(',')]

    return inspection_files

@click.command()
@click.argument('path')
@click.option("--dry-run", is_flag=True, help="Dont analyze inspections, only check if inspection metadata is provided. It's recommended to run this before analyzing a bunch of inspections as otherwise user interaction to provide metadata on the CLI will be required for each new inspection.")
@click.option("--metadata", is_flag=True, help="Always aks for metadata even if it is available. To correct errors made when providing it before.")
def analyze(path, dry_run, metadata):
    """Analyze inspection video(s) and store data to neo4j. Does not analyze simiarities and clusters, this is done with the command simiarities.

PATH Path to the video files. All subfolders will be searched. Inspections need to have an .mp4 video file and a .ass metadata file with the same name."""
    os.environ["OPENCV_LOG_LEVEL"]="SILENT"
    from pipeline import store_inspection
    from video_input.file_input import VideoFileFinder
    vi = VideoFileFinder(path)
    inspection_files = ask_for_selection(vi)
    for inspection_file in inspection_files:
        inspection = vi.get_inspection(inspection_file, ask_for_metadata=metadata)
        if not dry_run: store_inspection(inspection)


def ask_for_stored_inspection_ids():
    stored_inspections_dict = inspection_access.list()
    for i,(k, id) in enumerate(stored_inspections_dict.items()):
        print(f"{i:<3}: {k} with id {id}")
    
    selection = input("Select one or many videos (comma seperated). Type 'a' for all videos.")
    selected_inspection_keys = [k for i,k in enumerate(stored_inspections_dict) if 'a' == selection or str(i) in selection.replace(" ","").split(",")]
    selected_inspection_ids = [str(i) for k, i in stored_inspections_dict.items() if k in selected_inspection_keys]
    return selected_inspection_ids

@click.command()
def similarities():
    """Analyze previously stored inspections for similarities and clusters."""
    selected_inspection_ids = ask_for_stored_inspection_ids()

    if not selected_inspection_ids:
        return

    similarity_pipeline.do_similarity(inspection_filter=selected_inspection_ids)

@click.command()
def stitcher_demo():
    """Demonstration program of the image stitcher. Does not work via SSH as a X-Server is required."""
    demo()


messages.add_command(stitcher_demo)
messages.add_command(list)
messages.add_command(clear)
messages.add_command(analyze)
messages.add_command(similarities)
if __name__ == '__main__':
    messages()