FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

RUN groupadd -r default && useradd -r -g default chappie

ENV HOME=/home/chappie/app
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV TZ=Asia/Kolkata
ENV DEBIAN_FRONTEND=noninteractive


WORKDIR $HOME

RUN apt-get update && apt-get install -y \
    python3.11 python3-pip \
    nano gunicorn git 


RUN ln -s /usr/bin/python3 /usr/bin/python


# # -----------------------------------
# # create required folder
RUN mkdir $HOME/deepface

# # -----------------------------------
# # Copy required files from repo into image
COPY ./deepface $HOME/deepface
COPY ./api/app.py $HOME
COPY ./api/api.py $HOME
COPY ./api/routes.py $HOME
COPY ./api/service.py $HOME
COPY ./requirements.txt $HOME
COPY ./setup.py $HOME
COPY ./README.md $HOME


RUN python -m pip install --upgrade pip

# # update image os
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 libgl1 -y


# RUN chown -R chappie:default $HOME
# USER chappie


# # -----------------------------------
# # if you plan to use a GPU, you should install the 'tensorflow-gpu' package
# RUN pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host=files.pythonhosted.org tensorflow

# # -----------------------------------
# # install deepface from pypi release (might be out-of-date)
# # RUN pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host=files.pythonhosted.org deepface
# # -----------------------------------
# # install deepface from source code (always up-to-date)


RUN pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host=files.pythonhosted.org -e .

# # -----------------------------------
# # some packages are optional in deepface. activate if your task depends on one.
# # RUN pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host=files.pythonhosted.org cmake==3.24.1.1
# # RUN pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host=files.pythonhosted.org dlib==19.20.0
# # RUN pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host=files.pythonhosted.org lightgbm==2.3.1

# # RUN pip cache purge && rm $(basename $TF_URL)

# # -----------------------------------
# # run the app (re-configure port if necessary)
EXPOSE 5000
CMD ["gunicorn", "--workers=1", "--timeout=3600", "--bind=0.0.0.0:5000", "app:create_app()"]
