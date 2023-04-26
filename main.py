from norfair import Detection, Tracker
from typing import List
import numpy as np
import cv2
import torch
import argparse
from pymongo import MongoClient
from tqdm import tqdm

# connect db
HOST = 'watermelon3.inalab.net:27017'
USER = 'kth'
PWD = 'th28620720!'
client = MongoClient(
    HOST,
    username = USER,
    password = PWD
)
DB = client.urp_reboot


# ## ===== ===== ===== ===== ===== ===== ===== =====
# ## Parse arguments
# ## ===== ===== ===== ===== ===== ===== ===== =====

parser = argparse.ArgumentParser(description = "Object Tracking")
## Tracker
parser.add_argument('--threshold',  type=float,   default=0.7, help="Defines what is the maximum distance that can constitute a match")

## Video path
parser.add_argument('--video', type=str, default='/home/kth/URP/video_data/4KRoadtraffic.webm', help='video absolute path')

## Clusters
# parser.add_argument('--clusters',   type=str,   default='clusters', help='Name of the collection where store characteristic of clusters')
parser.add_argument('--objects',    type=str,   default='objects', help='Name of the collection where store all objects')

## GPU
parser.add_argument('--gpu',            type=int,   default=9,      help='GPU index')

args = parser.parse_args()

def yolo_detections_to_norfair_detections(
    yolo_detections: torch.tensor, track_points: str = "bbox"  # bbox or centroid
) -> List[Detection]:
    """convert detections_as_xywh to norfair detections"""
    norfair_detections: List[Detection] = []

    if track_points == "centroid":
        detections_as_xywh = yolo_detections.xywh[0]
        for detection_as_xywh in detections_as_xywh:
            centroid = np.array(
                [detection_as_xywh[0].item(), detection_as_xywh[1].item()]
            )
            scores = np.array([detection_as_xywh[4].item()])
            norfair_detections.append(
                Detection(
                    points=centroid,
                    scores=scores,
                    label=int(detection_as_xywh[-1].item()),
                )
            )
    elif track_points == "bbox":
        detections_as_xyxy = yolo_detections.xyxy[0]
        for detection_as_xyxy in detections_as_xyxy:
            bbox = np.array(
                [
                    [detection_as_xyxy[0].item(), detection_as_xyxy[1].item()],
                    [detection_as_xyxy[2].item(), detection_as_xyxy[3].item()],
                ]
            )
            scores = np.array(
                [detection_as_xyxy[4].item(), detection_as_xyxy[4].item()]
            )
            norfair_detections.append(
                Detection(
                    points=bbox, scores=scores, label=int(detection_as_xyxy[-1].item())
                )
            )

    return norfair_detections


if __name__ == '__main__':
    # Get collection name
    # coll_clusters = DB[args.clusters]
    coll_objects = DB[args.objects]

    # Change the device to GPU
    cuda_num = args.gpu
    device = torch.device(f'cuda:{cuda_num}' if torch.cuda.is_available() else "cpu")
    print(f'current using machine: {device}')

    # Set up a detector
    detector = torch.hub.load('ultralytics/yolov5', 'yolov5l')
    
    # Set up a video object
    path = args.video
    cap = cv2.VideoCapture(path)

    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print('About video object')
    print("length :", length)
    print("width :", width)
    print("height :", height)
    print("fps :", fps)

    # Set up a tracker
    tracker = Tracker(distance_function="iou", distance_threshold=args.threshold)

    with tqdm(total = length) as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            # if(int(cap.get(1)) % fps == 0):
            
            frame_num = cap.get(1)
            height, width, _ = frame.shape
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            yolo_detections = detector(frame)
            detections = yolo_detections_to_norfair_detections(
                    yolo_detections, track_points='bbox'
                )
            tracked_objects = tracker.update(detections=detections)
            for tracked_object in tracked_objects:
                object_ = {
                    'frame_id': frame_num,
                    # `[[x_min, y_min], [x_max, y_max]]`
                    'bbox': tracked_object.estimate.astype(int).tolist(),
                    'object_id': tracked_object.global_id
                    #class?
                }

                coll_objects.insert_one(object_)
            
            pbar.update(1)
                
