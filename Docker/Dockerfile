FROM r8.im/deforum-art/deforum-stable-diffusion@sha256:1a98303504c7d866d2b198bae0b03237eab82edc1491a5306895d12b0021d6f6

COPY huggingface /root/.cache/huggingface

RUN pip install boto3==1.26.143
RUN pip install cog==0.7.2
RUN pip install runpod==0.9.10

COPY src src
COPY predict_custom.py predict.py
COPY handler.py handler.py
ADD start.sh start.sh
RUN chmod +x start.sh

CMD ["./start.sh"]