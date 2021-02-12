FROM continuumio/anaconda3
COPY ./ml_rf /usr/local/python
EXPOSE 5000
WORKDIR /usr/local/python
RUN pip3 install -r requirements.txt
CMD python ml_rf_api.py

