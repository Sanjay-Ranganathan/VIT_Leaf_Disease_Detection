FROM python:3.10-slim AS builder

WORKDIR /app
RUN pip install --no-cache-dir --prefix=/install \
    torch==2.2.0+cpu torchvision==0.17.0+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html \
    streamlit Pillow timm numpy

FROM python:3.10-slim
WORKDIR /app
COPY --from=builder /install /usr/local
COPY . /app

EXPOSE 8501
CMD [ "streamlit","run","app.py","--server.port=8501","--server.address=0.0.0.0" ]