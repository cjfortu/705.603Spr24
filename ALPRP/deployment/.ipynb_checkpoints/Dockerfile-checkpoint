FROM python:3.8
COPY . ./
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 tesseract-ocr libtesseract-dev -y
RUN pip3 install -r requirements.txt
EXPOSE 23000 8785
CMD ["deployment_udp_client.py", "--host=127.0.0.1"]
ENTRYPOINT ["python"]