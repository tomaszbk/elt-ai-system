from python:3.11.6

# Install Python dependencies
COPY requirements.txt /requirements.txt

RUN pip install --no-cache-dir -r /requirements.txt

COPY . .

CMD python -m train