## Developed using Python 3.11.9
FROM python:3.11-slim

WORKDIR /usr/src/app

## Update tools
RUN pip install --upgrade pip setuptools wheel

## Copy the requirements
COPY requirements.txt ./

## Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

## Copy other app files
COPY . .

## Run the application
# CMD ["python", "./src/app.py"] # requires modification to the original file
CMD ["gunicorn", "--chdir", "src", "--bind", "0.0.0.0:8050", "app:server"]