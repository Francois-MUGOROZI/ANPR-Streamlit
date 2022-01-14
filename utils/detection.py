from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
import cv2
import easyocr

model_weights_path = "data/model_final_lg.pth"
config_file_path = "data/config_lg.yaml"

# Set up config
cfg = get_cfg()
cfg.merge_from_file(config_file_path)
cfg.MODEL.WEIGHTS = model_weights_path

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
# cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.6

predictor = DefaultPredictor(cfg)

labels = ['Plate', 'Licence', 'Vehicle']


def extract_licence_number(image, licence_box):
    reader = easyocr.Reader(['en'])
    text = reader.readtext(image[int(licence_box[1]):int(licence_box[0]), int(licence_box[3]):int(licence_box[2])])
    licence = text[0][1]
    return licence


def detection_from_image(image_path):
    image = cv2.imread(image_path)
    pred_class, scores, boxes = make_prediction(image)
    for i in range(len(pred_class)):
        box = boxes[i]
        score = scores[i]
        label = labels[pred_class[i]]
        licence = ''
        if label == 'Licence':
            licence = extract_licence_number(image, box)
        cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
        cv2.putText(image, f'{label}: {licence} {"{0:.2f}".format(score * 100)}%', (int(box[0]), int(box[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (232, 134, 16), 2)
    cv2.imshow("Detection from image", image)


def real_time_detection():
    cap = cv2.VideoCapture(0)
    while True:
        try:
            _, frame = cap.read()
            labels = ['Plate', 'Licence', 'Vehicle']
            pred_class, scores, boxes = make_prediction(frame)
            for i in range(len(pred_class)):
                box = boxes[i]
                score = scores[i]
                label = labels[pred_class[i]]
                licence = ''
                if label == 'Licence':
                    licence = extract_licence_number(frame, box)
                cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
                cv2.putText(frame, f'{label}: {licence}: {"{0:.2f}".format(score * 100)}%',
                            (int(box[0]), int(box[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.75, (232, 134, 16), 2)
            k = cv2.waitKey(10) & 0xff  # Press 'ESC' for exiting video
            cv2.imshow("Real time detection", frame)
            if k == 27:
                break  # Do a bit of cleanup
        except:
            pass
    cap.release()
    cv2.destroyAllWindows()


def make_prediction(image):
    outputs = predictor(image)
    instances = outputs['instances']
    boxes = list(instances.pred_boxes)
    pred_class = list(instances.pred_classes)
    scores = list(instances.scores)
    return pred_class, scores, boxes


def draw_detection_result(pred_class, scores, boxes, image):
    licences = []
    for i in range(0, len(pred_class) - 1):
        box = boxes[i]
        score = scores[i]
        label = labels[pred_class[i]]
        licence = ''
        print(label)
        if label == 'Licence':
            licence = extract_licence_number(image, box)
            licences.append(licence)
        cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
        cv2.putText(image, f'{label}: {licence} {"{0:.2f}".format(score * 100)}%', (int(box[0]), int(box[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (232, 134, 16), 2)
    return image, licences
