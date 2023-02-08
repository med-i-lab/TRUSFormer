from pyroi.segmentation_server import SegmentationServer, SegmentationBackend
from pyroi.backend.backend_example import DummySegmentations
from src.data.exact.segmentation_app import ExactVuSegmentationBackend

with open("cores_list.txt") as f:
    cores_list = f.read().split()

backend = ExactVuSegmentationBackend(
    cores_list,
)

app = SegmentationServer(backend).app
