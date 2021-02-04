import sys
import os
import cv2
from keras.models import load_model
import numpy as np
import yaml

from utils.datasets import get_labels
from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.inference import load_image
from utils.preprocessor import preprocess_input
from PIL import Image

def is_image(path):
    try:
        im = Image.open(path)
        return True
    except IOError:
        return False

def save_results(preds,img_name,img_path,config):
    pipeline_dir = config['general']['pipeline_path']
    results_filename = config['face_classification']['results']
    docker_workdir = config['face_classification']['workdir']
    cleaned_results = preds
    # for item in decode_predictions(preds,top=5)[0]:
    #     cleaned_results[str(item[1])] = float(item[2])
    # print(cleaned_results)
    img_name = remove_file_extension(img_name)
    print(img_name)
    img_dir_wo_parent = get_path_wo_top_parent_dirs(img_path)
    print(img_dir_wo_parent)
    pipeline_path = os.path.join(pipeline_dir,img_dir_wo_parent,img_name)
    print(pipeline_path)
    results_filepath = os.path.join(docker_workdir,pipeline_path,results_filename)
    print(results_filepath)
    print('********************************')
    yaml.dump(cleaned_results, open(results_filepath, "w"), default_flow_style=False)
    return None

def remove_file_extension(file_name):
    return '.'.join(file_name.split('.')[:-1])

def get_path_wo_top_parent_dirs(file_path):
    # return file_path
    return '/'.join(file_path.split('/')[-3:])


# parameters for loading data and images
# image_path = sys.argv[1]
input_dir =  sys.argv[1]
config_file = sys.argv[2]
with open(config_file, 'r') as stream:
    config = yaml.safe_load(stream)
print('Performing face detection for',input_dir)
# results_dir = sys.argv[2]
# detection_model_path = '../trained_models/detection_models/haarcascade_frontalface_default.xml'
# emotion_model_path = '../trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5'
# gender_model_path = '../trained_models/gender_models/simple_CNN.81-0.96.hdf5'
detection_model_path = config['face_classification']['detection_model_path']
emotion_model_path = config['face_classification']['emotion_model_path']
gender_model_path = config['face_classification']['gender_model_path']

emotion_labels = get_labels('fer2013')
gender_labels = get_labels('imdb')
font = cv2.FONT_HERSHEY_SIMPLEX

# hyper-parameters for bounding boxes shape
gender_offsets = (30, 60)
gender_offsets = (10, 10)
emotion_offsets = (20, 40)
emotion_offsets = (0, 0)

# loading models
face_detection = load_detection_model(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)
gender_classifier = load_model(gender_model_path, compile=False)

# getting input model shapes for inference
emotion_target_size = emotion_classifier.input_shape[1:3]
gender_target_size = gender_classifier.input_shape[1:3]



for image in os.listdir(input_dir):

    # loading images
    image_path = os.path.join(input_dir,image)
    print(image_path,':', is_image(image_path))
    if not is_image(image_path):
        continue
    rgb_image = load_image(image_path, grayscale=False)
    gray_image = load_image(image_path, grayscale=True)
    gray_image = np.squeeze(gray_image)
    gray_image = gray_image.astype('uint8')

    faces = detect_faces(face_detection, gray_image)
    results = {}
    for i,face_coordinates in enumerate(faces):
        x1, x2, y1, y2 = apply_offsets(face_coordinates, gender_offsets)
        rgb_face = rgb_image[y1:y2, x1:x2]

        x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
        gray_face = gray_image[y1:y2, x1:x2]

        try:
            rgb_face = cv2.resize(rgb_face, (gender_target_size))
            gray_face = cv2.resize(gray_face, (emotion_target_size))
        except:
            continue

        rgb_face = preprocess_input(rgb_face, False)
        rgb_face = np.expand_dims(rgb_face, 0)
        gender_prediction = gender_classifier.predict(rgb_face)
        gender_label_arg = np.argmax(gender_prediction)
        gender_text = gender_labels[gender_label_arg]

        gray_face = preprocess_input(gray_face, True)
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)
        emotion_label_arg = np.argmax(emotion_classifier.predict(gray_face))
        emotion_text = emotion_labels[emotion_label_arg]

        if gender_text == gender_labels[0]:
            color = (0, 0, 255)
        else:
            color = (255, 0, 0)

        bounding_box_str = str(x1) +', ' + str(x2) +', ' + str(y1) +', ' + str(y2)
        results['person_' + str(i+1)] = {'gender': gender_text,
                                         'emotion': emotion_text,
                                         'bounding_box': bounding_box_str}
    print(results)
    save_results(results,image,input_dir,config)
