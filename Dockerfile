FROM nialljb/cuda-base:latest
# FROM nialljb/cuda-mamba:latest

# Setup environment for Docker image
ENV HOME=/root/
ENV FLYWHEEL="/flywheel/v0"
WORKDIR $FLYWHEEL
RUN mkdir -p $FLYWHEEL/input

# Ensure pip is upgraded and install `packaging`
RUN pip install --no-cache-dir --upgrade pip setuptools wheel packaging

# Install PyTorch separately before requirements.txt
RUN pip install --no-cache-dir torch==2.2.2 torchvision==0.17.2

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the contents of the directory the Dockerfile is into the working directory of the to be container
COPY ./ $FLYWHEEL/

# Configure entrypoint
RUN bash -c 'chmod +rx $FLYWHEEL/run.py' && \
    bash -c 'chmod +rx $FLYWHEEL/app/'

ENTRYPOINT ["python", "/flywheel/v0/run.py"] 