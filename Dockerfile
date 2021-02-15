FROM python:3.7

WORKDIR /portfolio

COPY requirements.txt .

# Install dependencies
RUN pip install -r requirements.txt

# Copy python files
COPY app.py .
COPY garbage_app.py .
COPY grammar_app.py .

#Copy models and weights
COPY garbage_model_weights.h5 .
COPY garbage_model.json .
COPY grammar_model.pt .

COPY ./static ./static
COPY ./templates ./templates


CMD ["python", "app.py"]