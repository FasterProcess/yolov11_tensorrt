import numpy as np
from typing import Tuple
import cv2
import os


class CalibratorDataset:
    """
    you must write your own class refer to your model preprocess

    _______________________
    must init:

    self.datasets: List[item], item is model input, such as np.ndarray with shape NCHW
    self.datasize: is len(self.datasets)
    """

    def __init__(
        self,
        calibration_image_folder,
        input_shape=(-1, 3, 640, 640),
        dataset_limit=300,
        skip_frame=20,
        batch_size=1,
    ):
        self.image_folder = calibration_image_folder

        self.preprocess_flag = True
        self.datasets = None
        self.datasize = 0

        self.dataset_limit = dataset_limit
        self.skip_frame = skip_frame

        (_, _, self.height, self.width) = input_shape
        self.batch_size = batch_size

        self.init_data()

    def data(self):
        return self.datasets

    def __len__(self) -> int:
        if self.datasets is None:
            return 0
        return len(self.datasets)

    def shape(self) -> tuple:
        return self.datasets[0].shape

    def __getitem__(self, index) -> np.ndarray:
        if index < self.datasize:
            return self.datasets[index]
        else:
            return None

    def init_data(self):
        self.preprocess_flag = False
        self.datasets = self.load_pre_data(
            self.image_folder, size_limit=self.dataset_limit, skip=self.skip_frame
        )  # (k*b+m,c,h,w)
        self.datasize = (
            len(self.datasets) // self.batch_size * self.batch_size // self.batch_size
        )

        self.datasets = np.split(
            self.datasets[: self.datasize * self.batch_size, ...],
            self.datasize,
            axis=0,
        )

        self.datasets = [np.ascontiguousarray(data) for data in self.datasets]
        print(
            f"finish init calibration in cpu: datasize={len(self)}*{self.shape()}, type={self.datasets[0].dtype}"
        )

    def letterbox(
        self, img: np.ndarray, new_shape: Tuple[int, int] = (640, 640)
    ) -> Tuple[np.ndarray, Tuple[int, int]]:
        shape = img.shape[:2]  # current shape [height, width]

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = (new_shape[1] - new_unpad[0]) / 2, (
            new_shape[0] - new_unpad[1]
        ) / 2  # wh padding

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )

        return img, (top, left)

    def preprocess(self, imgs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess the input image before performing inference.

        This method reads the input image, converts its color space, applies letterboxing to maintain aspect ratio,
        normalizes pixel values, and prepares the image data for model input.

        Returns:
            image_datas (np.ndarray): Preprocessed image data ready for inference with shape (B, 3, height, width).
        """
        # Convert the image color space from BGR to RGB
        image_datas = []
        for i in range(len(imgs)):
            img = imgs[i]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img, _ = self.letterbox(img, (self.width, self.height))
            image_data = np.transpose(img, (2, 0, 1))  # Channel first
            image_data = np.array(image_data) / 255.0
            image_datas.append(image_data)
        return np.stack(image_datas, dtype=np.float32)

    def load_pre_data(self, videos_folder, size_limit=0, skip=20):
        videos = os.listdir(videos_folder)
        imgs = []

        for video in videos:
            video_path = os.path.join(videos_folder, video)
            cap = cv2.VideoCapture(video_path)
            idx = -1
            while True:
                ret, frame = cap.read()  # ret表示是否成功读取，frame是当前帧
                idx += 1
                if not ret:
                    break

                if idx % skip == 0:
                    print(f"load {video_path}: {idx}")
                    imgs.append(frame)

                if size_limit > 0 and len(imgs) >= size_limit:
                    break
            cap.release()
            if size_limit > 0 and len(imgs) >= size_limit:
                break

        assert len(imgs) > 0, "empty datas"

        imgs = np.stack(imgs)

        return self.preprocess(imgs)
