from detectron2.utils.logger import setup_logger
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
import cv2
import easyocr

setup_logger()

model_weights_path = "data/model_final.pth"
config_file_path = "data/config.yaml"

# Set up config
cfg = get_cfg()
cfg.merge_from_file(config_file_path)
cfg.MODEL.WEIGHTS = model_weights_path
labels = ['Plate', 'Licence', 'Vehicle']


def extract_licence_number(image, licence_box):
    reader = easyocr.Reader(['en'])
    text = reader.readtext(image[int(licence_box[1]):int(licence_box[0]), int(licence_box[3]):int(licence_box[2])])
    licence = text[0][1]
    return licence


def detection_from_image(image_path, threshold):
    image = cv2.imread(image_path)
    pred_class, scores, boxes = make_prediction(image, threshold)
    for i in range(len(pred_class)):
        box = boxes[i]
        score = scores[i]
        label = labels[pred_class[i]]
        print(label)
        licence = ''
        if label == 'Licence':
            licence = extract_licence_number(image, box)
            print(licence)
        cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
        cv2.putText(image, f'{label}: {licence}: {"{0:.2f}".format(score * 100)}%', (int(box[0]), int(box[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (232, 134, 16), 2)
    cv2.imshow("Detection from image", image)


def make_prediction(image, threshold):
    # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = threshold

    # Set up a predictor
    predictor = DefaultPredictor(cfg)
    outputs = predictor(image)

    instances = outputs['instances']
    boxes = list(instances.pred_boxes)
    pred_class = list(instances.pred_classes)
    scores = list(instances.scores)
    return pred_class, scores, boxes