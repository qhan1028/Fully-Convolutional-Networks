FROM tensorflow/tensorflow:latest-gpu

MAINTAINER Qhan <qhan@ailabs.tw>

## -----------------------------------------------------------------------------
## Install libraries for brakground replacer

RUN apt-get update
RUN apt-get install -y \
        python-tk \
        vim \ 
        tree \ 
        htop \
        python-opencv \
        python3-pip

RUN pip3 install \
        Cython \
        easydict==1.6 \
        hickle \
        pyyaml \
        numpy>=1.12.0 \
        Pillow>=3.4.2 \
        protobuf>=3.2.0 \
        scikit-image>=0.12.3 \
        scikit-learn>=0.17.1 \
        scipy>=0.17.1 \
        six>=1.10.0 \
        tensorflow-gpu==1.2.1 \
        opencv-python


## -----------------------------------------------------------------------------
## Load source code

# Set the working directory to /app
WORKDIR /app

# Add vim profile
COPY .vimrc /root/


CMD ["bash"]
