FROM python:3.8
COPY . ./
RUN pip3 install -r requirements.txt
EXPOSE 8786
CMD ["ml_Service_Template.py", "--host=0.0.0.0"]
ENTRYPOINT ["python"]