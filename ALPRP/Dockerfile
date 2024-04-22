FROM python:3.8
COPY . ./
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 tesseract-ocr libtesseract-dev -y
RUN pip3 install -r requirements.txt
EXPOSE 23000 8785
CMD ["service.py", "--host=0.0.0.0"]
ENTRYPOINT ["python"]