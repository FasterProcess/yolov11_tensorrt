from yolo import YOLOv8_11Trt as YOLO
import cv2
import numpy as np

yolo = YOLO(
    "model/yolov11m_dynamic.engine",
    confidence_thres=0.5,
    iou_thres=0.5,
    max_batch_size=1,
)

cap = cv2.VideoCapture("dataset/example.mp4")
out_cap = cv2.VideoWriter(
            "dataset/output.mp4",
            cv2.VideoWriter_fourcc(*"mp4v"),
            cap.get(cv2.CAP_PROP_FPS),
            (
                int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            ),
        )

while True:
    ret,frame = cap.read()
    if not ret:
        break

    batch_frame=np.stack([frame])
    
    # run async
    batch_results = yolo.detect(batch_frame)
    
    # # run sync
    # yolo.detect_sync(batch_frame)             # commit task
    # batch_frame,batch_results = yolo.detect_sync_output(wait=True)    # read result
    
    result_frames =batch_frame.copy()
    yolo.draw_detections(result_frames, batch_results)
    
    for result_frame in result_frames:
        out_cap.write(result_frame)
                
cap.release()
out_cap.release()
yolo.release()