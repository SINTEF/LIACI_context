from typing import List
import click
import sys
from computer_vision.LIACI_stitcher import demo
from video_input.file_input import VideoFileFinder
from video_input.inspection_video_input import InspectionVideoFile
import similarity_pipeline
import scree_plots.scree_plot_pca

import data.access.inspection as inspection_access



@click.group()
def messages():
    pass

@click.command()
@click.argument('path')
def list(path):
    from video_input.file_input import VideoFileFinder
    vi = VideoFileFinder(path)
    for i, inspection_file in enumerate(vi.list_inspections()):
        print(f"{i:<2}: {inspection_file.video_file: <70}, {inspection_file.video_file_size/1024/1024:12.2f}MB video, {inspection_file.context_file_size/1024/1024:12.2f}MB context")

@click.command()
@click.argument('what', type=click.Choice(['similarities', 'nodes']))
def clear(what):
    if 'similarities' == what:
        clear_similarities()
    if 'nodes' == what:
        clear_nodes()


def clear_nodes():
    from data.access.datastore import delete_all_neo4j_database
    if input("DELETE DATABASE? (Canot be undone) Y/N ") == "Y":
        delete_all_neo4j_database()
        print("database deleted.")
    else:
        print("database NOT deleted.")

def clear_similarities():
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
@click.option("--dry-run", is_flag=True)
@click.option("--metadata", is_flag=True)
def analyze(path, dry_run, metadata):
    from pipeline import store_inspection
    from video_input.file_input import VideoFileFinder
    vi = VideoFileFinder(path)
    inspection_files = ask_for_selection(vi)
    for inspection_file in inspection_files:
        inspection = vi.get_inspection(inspection_file, ask_for_metadata=metadata)
        if not dry_run: store_inspection(inspection)


def ask_for_stored_inspection_ids():
    stored_inspections_dict = inspection_access.list()
    for i,k in enumerate(stored_inspections_dict):
        print(f"{i:<3}: {k}")
    
    selection = input("Select one or many videos (comma seperated). Type 'a' for all videos.")
    selected_inspection_keys = [k for i,k in enumerate(stored_inspections_dict) if 'a' == selection or str(i) in selection.replace(" ","").split(",")]
    selected_inspection_ids = [str(i) for k, i in stored_inspections_dict.items() if k in selected_inspection_keys]
    return selected_inspection_ids

@click.command()
def similarities():
    selected_inspection_ids = ask_for_stored_inspection_ids()

    if not selected_inspection_ids:
        return

    similarity_pipeline.do_similarity(inspection_filter=selected_inspection_ids)

@click.command()
def stitcher_demo():
    demo()


messages.add_command(stitcher_demo)
messages.add_command(list)
messages.add_command(clear)
messages.add_command(analyze)
messages.add_command(similarities)
if __name__ == '__main__':
    messages()