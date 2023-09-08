from ultralytics import YOLO

class DetectShips:
    def __init__(self, best_model):
        self.best_model = best_model
        self.model = YOLO(self.best_model)

    def infer(self, image, conf=0.65, iou=0.7):
        results = self.model.track(source=image, device=device, conf=conf, iou=iou)[0]
        classes = results.names
        annotations = results.boxes.data.numpy.tolist()
        bboxes = []
        for xmin, ymin, xmax, ymax, i, p, c in annotations:
            bboxes.append([int(xmin), int(ymin), int(xmax), int(ymax), int(id), round(float(p * 100), 2), classes[int(c)]])
        return bboxes

if __name__ == '__main__':
    model = DetectShips(r'path\to\best.pt')
    model.detect(image)
