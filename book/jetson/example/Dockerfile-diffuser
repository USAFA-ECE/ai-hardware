FROM nvcr.io/nvidia/pytorch:23.09-py3

EXPOSE 5858

WORKDIR /app
COPY ./generate_img.py /app

# Ideally, pin to versions you know work.
RUN pip install --no-cache-dir \
         diffusers[torch] \
         fastapi \
         transformers \
         uvicorn

CMD ["uvicorn", "generate_img:app", "--host", "0.0.0.0", "--port", "5858"]
