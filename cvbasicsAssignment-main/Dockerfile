FROM python:3.8
COPY . ./
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN pip3 install -r requirements.txt
EXPOSE 8786
CMD ["objectDetectionService.py", "--host=0.0.0.0"]
ENTRYPOINT ["python"]