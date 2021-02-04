
FROM nvidia/cuda:10.1-base-ubuntu16.04
RUN apt-get -y update && apt-get install -y git python3-pip python3-dev python3-tk vim procps curl
RUN apt update
RUN apt-get install -y libglib2.0-0 libsm6 libxrender1 libxext6
#Face classificarion dependencies & web application
RUN python3 -m pip install  --upgrade "pip < 21.0"
RUN python3 -m pip install numpy
RUN python3 -m pip install scipy scikit-learn pillow tensorflow pandas h5py
RUN python3 -m pip install opencv-python==3.4.2.17 keras statistics pyyaml pyparsing cycler matplotlib Flask
RUN python3 -m pip install scikit-image imageio

ADD . /ekholabs/face-classifier

WORKDIR ekholabs/face-classifier/src

#ENV PYTHONPATH=$PYTHONPATH:src
#ENV FACE_CLASSIFIER_PORT=8084
#EXPOSE $FACE_CLASSIFIER_PORT

#ENTRYPOINT ["python3"]
#CMD ["src/web/faces.py"]
