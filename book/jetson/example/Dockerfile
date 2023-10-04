FROM nvcr.io/nvidia/l4t-tensorflow:r35.3.1-tf2.11-ру3

WORKDIR /app
COPY ./tf_diffusion.py /app/tf_diffusion.py

RUN pip install upgrade --no-cache-dir \
	--extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v512 \
	tensorflow==2.12.0+nv23.06

RUN pip install --upgrade --no-cache-dir \
	pillow \
	keras_CV==0.6.1 \
	keras-core
