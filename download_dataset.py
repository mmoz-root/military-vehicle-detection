import os
from dotenv import load_dotenv
from roboflow import Roboflow

load_dotenv()

rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])
project = rf.workspace("muhammed-hocy2").project("military-vehicle-detection-juleg-x1b34")
dataset = project.version(1).download("yolov8", location="dataset/")

print(f"\nDataset downloaded to: {dataset.location}")
