FROM python:3.8
COPY . ./
RUN pip3 install -r requirements.txt
EXPOSE 8785
CMD ["carfactors_service.py", "--host=0.0.0.0"]
ENTRYPOINT ["python"]