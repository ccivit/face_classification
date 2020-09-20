FROM nvidia/cuda:9.0-cudnn7-runtime-ubuntu16.04
RUN apt-get update && \
        apt-get install -y software-properties-common && \
        add-apt-repository ppa:deadsnakes/ppa && \
        apt-get update -y  && \
        apt-get install -y build-essential python3.6 python3.6-dev python3-pip && \
        apt-get install -y git nano procps curl vim
        # update pip
RUN python3.6 -m pip install pip --upgrade && \
    python3.6 -m pip install numpy scipy scikit-learn pillow tensorflow pandas h5py opencv-python==3.2.0.8 keras statistics pyyaml pyparsing cycler matplotlib Flask
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.6 1

WORKDIR /data_project

ENTRYPOINT ["/bin/bash"]



#RUN apt-get -y update && apt-get install -y git python3-pip python3-dev python3-tk vim procps curl

#Face classificarion dependencies & web application
#RUN pip3 install numpy scipy scikit-learn pillow tensorflow pandas h5py opencv-python==3.2.0.8 keras statistics pyyaml pyparsing cycler matplotlib Flask

#ADD . /ekholabs/face-classifier

#WORKDIR ekholabs/face-classifier

#ENV PYTHONPATH=$PYTHONPATH:src
#ENV FACE_CLASSIFIER_PORT=8084
#EXPOSE $FACE_CLASSIFIER_PORT

#ENTRYPOINT ["python3"]
#CMD ["src/web/faces.py"]
