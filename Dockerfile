FROM python:3.11.10-slim

RUN pip install pipenv

# Set the working directory inside the container
WORKDIR /app

COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy

COPY ["predict.py", "predict_test.py", "train.py", "model.bin", "./"]

# Expose the port the app runs on
EXPOSE 9696

# Define the command to run the application using gunicorn
ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict:app"]