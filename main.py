
from pipeline.pipeline import Pipeline
from pipeline.video_input.file_input import VideoFileInput
import data.datastore as datastore


vi = VideoFileInput('../')
for i, inspection in enumerate(vi.list_inspections()):
    print(f"{i}: {inspection['video_file']: <60}, {inspection['video_file_size']/1024/1024:12.2f}MB video, {inspection['context_file_size']/1024/1024:12.2f}MB context")

if input("DELETE DATABASE? (Canot be undone) Y/N ") == "Y":
    datastore.delete_all_neo4j_database()
    print("database deleted.")
else:
    print("database NOT deleted.")

pipeline = Pipeline()
print("all inspection read")

for inspection in vi.read_inspections():
    #pipeline.prepare_metadata_for_preanalyzed_inspection(inspection)
    try:
        pipeline.store_inspection(inspection)
    except datastore.EntryDoesExistExeption as e:
        print(f"Will not store inspection {inspection.video_file}: {e}")


