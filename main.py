
from pipeline.pipeline import Pipeline
from pipeline.video_input.file_input import VideoFileInput
import data.datastore as datastore


vi = VideoFileInput('../')
for i, inspection in enumerate(vi.list_inspections()):
    print(f"{i}: {inspection['video_file']: <60}, {inspection['video_file_size']/1024/1024:12.2f}MB video, {inspection['context_file_size']/1024/1024:12.2f}MB context")

pipeline = Pipeline()
datastore.delete_all_neo4j_database()
for inspection in vi.read_inspections():
    #pipeline.prepare_metadata_for_preanalyzed_inspection(inspection)
    pipeline.store_preanalyzed_inspection(inspection)
    #exit(0)

print("all inspection read")
