FROM python:3.10-slim

COPY huggingface /root/.cache/huggingface
RUN apt update && apt install ffmpeg -y
COPY install_requirements.py install_requirements.py
RUN pip install runpod==0.9.10 && python install_requirements.py && pip cache purge

COPY deforum-stable-diffusion-main /deforum-stable-diffusion-main
WORKDIR /deforum-stable-diffusion-main

COPY handler.py handler.py
CMD ["python", "handler.py"]