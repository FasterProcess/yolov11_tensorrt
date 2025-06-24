# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
from typing import List, Tuple, Dict
import cv2
import numpy as np
import onnxruntime as ort
from threading import Thread
from queue import Queue
from detect_box import DetectBox
import argparse
import os
from yolo import YOLOv8_11Trt as YOLO


def write_result(output_video, detection: YOLO, fps, raw_width, raw_height):
    if output_video is None:
        out_cap = None
    else:
        out_cap = cv2.VideoWriter(
            output_video,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (
                raw_width,
                raw_height,
            ),
        )

    while True:
        data = detection.detect_sync_output(wait=True)
        if data is None:
            if out_cap is not None:
                print(f"save to {output_video}")
                out_cap.release()
            return
        else:
            imgs, detect_boxs = data
            for i in range(len(imgs)):
                print(
                    f"find person: {len(detect_boxs[i]['person']) if 'person' in detect_boxs[i] else 0}"
                )

            detection.draw_detections(imgs, detect_boxs)
            if out_cap is not None:
                for i in range(len(imgs)):
                    out_cap.write(imgs[i])


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="model/yolov11m_dynamic.engine",
        help="Input your tensorrt engine model.",
    )
    parser.add_argument(
        "--input_video",
        type=str,
        default="input/-rSB43WE_34_7_422.mp4",
        help="input_video",
    )
    parser.add_argument(
        "--output_video",
        type=str,
        default=None,
        help="output_video, none will no save file. auto means dynamic path",
    )
    parser.add_argument(
        "--batchsize",
        type=int,
        default=1,
        help="batchsize",
    )
    parser.add_argument(
        "--disable_pipeline",
        type=bool,
        default=False,
        help="disable pipeline parallel",
    )
    parser.add_argument(
        "--raw_input",
        type=bool,
        default=False,
        help="False will give torch.Tensor as preprocess input",
    )

    return parser.parse_args()


class FakeSource:
    """
    used to gen fake test input data
    """

    def __init__(self, input_shape, size):
        self.max_idx = size
        self.data = np.random.randint(low=0, high=255, size=input_shape, dtype=np.uint8)
        self.idx = -1

    def read(self) -> Tuple[bool, np.ndarray]:
        self.idx += 1
        if self.idx <= self.max_idx:
            return True, self.data.copy()
        else:
            return False, None

    def release(self):
        pass


def main_async(args):
    """
    run async demo
    """
    if args.output_video == "auto":
        output_video = args.input_video.replace(".mp4", "_gen.mp4")
    else:
        output_video = args.output_video

    MAX_BATCH = args.batchsize

    detection = YOLO(
        args.model,
        confidence_thres=0.5,
        iou_thres=0.5,
        max_batch_size=MAX_BATCH,
        force_torch_input=not args.raw_input,
    )
    cap = cv2.VideoCapture(args.input_video)

    if output_video is None:
        out_cap = cv2.VideoWriter(
            output_video,
            cv2.VideoWriter_fourcc(*"mp4v"),
            cap.get(cv2.CAP_PROP_FPS),
            (
                int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            ),
        )
    else:
        out_cap = None

    import time

    start_time = time.time()
    is_end = False

    while not is_end:
        imgs = []
        while True:
            ret, frame = cap.read()  # ret表示是否成功读取，frame是当前帧
            if not ret:
                is_end = True
                break

            imgs.append(frame)
            if len(imgs) >= MAX_BATCH:
                break
        if not imgs:
            break

        imgs = np.stack(imgs)
        detection.detect_sync(imgs)
        imgs, detect_boxs = detection.detect_sync_output(wait=True)
        for i in range(len(imgs)):
            print(
                f"find person: {len(detect_boxs[i]['person']) if 'person' in detect_boxs[i] else 0}"
            )

        detection.draw_detections(imgs, detect_boxs)
        if out_cap is not None:
            for i in range(len(imgs)):
                out_cap.write(imgs[i])
    end_time = time.time()
    cap.release()
    if out_cap is not None:
        out_cap.release()
    detection.release()

    print(f"time cost: {end_time-start_time:.1f}s")
    print(f"save to {output_video}")


def main_sync(args):
    """
    run sync demo
    """
    if args.output_video == "auto":
        output_video = args.input_video.replace(".mp4", "_gen.mp4")
    else:
        output_video = args.output_video
    MAX_BATCH = args.batchsize

    detection = YOLO(
        args.model,
        confidence_thres=0.5,
        iou_thres=0.5,
        max_batch_size=MAX_BATCH,
        force_torch_input=not args.raw_input,
    )
    cap = cv2.VideoCapture(args.input_video)

    write_thread = Thread(
        target=write_result,
        args=(
            output_video,
            detection,
            cap.get(cv2.CAP_PROP_FPS),
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        ),
    )
    write_thread.start()
    import time

    is_end = False

    # # test
    # cap.release()
    # cap = FakeSource((1080, 1920, 3), 1000)
    start_time = time.time()
    while not is_end:
        imgs = []
        while True:
            ret, frame = cap.read()  # ret表示是否成功读取，frame是当前帧
            if not ret:
                is_end = True
                break

            imgs.append(frame)
            if len(imgs) >= MAX_BATCH:
                break
        if not imgs:
            break

        imgs = np.stack(imgs)
        detection.detect_sync(imgs)

    detection.release()
    write_thread.join()
    end_time = time.time()
    print(f"time cost: {end_time-start_time:.1f}s")
    cap.release()


# test: python3 main.py --input_video=dataset/example.mp4 --model=model/yolov11m_dynamic.engine --output_video="auto"
if __name__ == "__main__":
    args = get_args()
    if args.disable_pipeline:
        main_async(args)  # run pre-infer-post one-by-one
    else:
        main_sync(args)  # pipeline parallel
