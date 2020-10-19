FROM repo.eresearch.unimelb.edu.au:8000/happypet/tensorflow:1.15.2-gpu-py3

WORKDIR /app

COPY ["./src", "requirements.txt", "./"]

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

CMD ["python", "app.py"]