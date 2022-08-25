
import sys
from similarity_pipeline import do_similarity
from scree_plots.scree_plot_pca import scree_plot_inspections

do_similarity()
# scree_plot_inspections()
exit(1)


from pipeline import Pipeline
from video_input.file_input import VideoFileInput
from data.access.datastore import delete_all_neo4j_database, EntryDoesExistExeption

vi = VideoFileInput('../../')
for i, inspection in enumerate(vi.list_inspections()):
    print(f"{i}: {inspection['video_file']: <60}, {inspection['video_file_size']/1024/1024:12.2f}MB video, {inspection['context_file_size']/1024/1024:12.2f}MB context")

if (len(sys.argv) and sys.argv[1] == '-d') or input("DELETE DATABASE? (Canot be undone) Y/N ") == "Y":
    delete_all_neo4j_database()
    print("database deleted.")
else:
    print("database NOT deleted.")

pipeline = Pipeline()
print("all inspection read")

for inspection in vi.read_inspections():
    #pipeline.prepare_metadata_for_preanalyzed_inspection(inspection)
    try:
        pipeline.store_inspection(inspection)
    except EntryDoesExistExeption as e:
        print(f"Will not store inspection {inspection.video_file}: {e}")


