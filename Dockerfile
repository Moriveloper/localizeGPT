FROM public.ecr.aws/lambda/python:3.8

# Upgrade pip
RUN python3 -m pip install --upgrade pip

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY handler.py ./

CMD ["handler.lambda_handler"]