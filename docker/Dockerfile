FROM nvidia/cuda:11.0-base
RUN apt-get update && \
        apt-get install -y software-properties-common && \
        add-apt-repository ppa:deadsnakes/ppa && \
        apt-get update -y  && \
        apt-get install -y build-essential python3.6 python3.6-dev python3-pip && \
        apt-get install -y git nano procps curl vim
        
        
RUN python3.6 -m pip install pip --upgrade && \
    python3.6 -m pip install numpy scipy==1.2.1 scikit-learn matplotlib cycler pyyaml pyparsing pandas
RUN python3.6 -m pip install pip --upgrade && \
    python3.6 -m pip install tensorflow h5py 
RUN python3.6 -m pip install opencv-python &&\ 
    apt update && apt install -y libsm6 libxext6 &&\
    apt-get install -y libxrender-dev  && \
    apt update && apt install -y libgl1-mesa-glx
    
RUN python3.6 -m pip install pip --upgrade && \
    python3.6 -m pip install keras statistics pillow imageio scikit-image
   
    
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.6 1

WORKDIR /data_project/src

ENTRYPOINT ["python", "image_emotion_gender_demo.py"]
