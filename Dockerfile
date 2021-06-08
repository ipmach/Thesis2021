FROM tensorflow/tensorflow:2.3.0-gpu

# copy the dependencies file to the working directory
ADD . /code/

RUN mkdir /res

# set the working directory in the container
WORKDIR /code

# Add graphviz
RUN apt-get update
RUN apt install graphviz --yes
RUN apt install vim --yes

# install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

## copy the content of the local src directory to the working directory
#COPY . .

## command to run on container start
#CMD ./train_models.sh
