FROM python:3.10.11
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir numpy
RUN pip install --no-cache-dir numba
CMD ["python3", "timestamp_matching.py"]