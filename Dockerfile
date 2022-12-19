FROM swr.cn-south-1.myhuaweicloud.com/mindspore/mindspore-gpu-cuda11.1:1.9.0
ENV workdir=/home/workdir
WORKDIR ${workdir}
COPY . ${workdir}
RUN pip install -r requriments.txt