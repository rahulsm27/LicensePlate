from super_gradients.training import models
from super_gradients.common.object_names import Models

from sort import *

model = models.get(Models.YOLO_NAS_S, pretrained_weights="coco")

video_path = 'testwalk.mp4'

predictions = model.predict(video_path)
predictions.show()
predictions.save("detections.mp4")