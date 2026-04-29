FROM nialljb/cuda-base:latest
# FROM nialljb/cuda-mamba:latest

# Setup environment for Docker image
ENV HOME=/root/
ENV FLYWHEEL="/flywheel/v0"
WORKDIR $FLYWHEEL
RUN mkdir -p $FLYWHEEL/input

# Ensure pip is upgraded and install `packaging`
RUN pip install --no-cache-dir --upgrade pip setuptools wheel packaging 

# Install the CUDA 12.1 PyTorch wheels directly to avoid downloading the large
# split nvidia-* dependency wheels from PyPI during image builds.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 torch==2.2.2 torchvision==0.17.2

# Install ANTs, added cmake
RUN apt-get update && apt-get install -y curl tar unzip git cmake && \
    curl -fsSL https://github.com/ANTsX/ANTs/releases/download/v2.5.4/ants-2.5.4-almalinux8-X64-gcc.zip -o /tmp/ants.tar.gz && \
    curl -fsSL https://github.com/rordenlab/dcm2niix/releases/latest/download/dcm2niix_lnx.zip -o /tmp/dcm2niix.zip && \
    unzip /tmp/ants.tar.gz -d /opt/ && \
    unzip /tmp/dcm2niix.zip -d /tmp/dcm2niix && \
    install -m 0755 /tmp/dcm2niix/dcm2niix /usr/local/bin/dcm2niix && \
    rm /tmp/ants.tar.gz && \
    rm -rf /tmp/dcm2niix.zip /tmp/dcm2niix && \
    echo 'export PATH=/opt/ants-2.5.4/bin:$PATH' >> ~/.bashrc

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --only-binary=SimpleITK SimpleITK
RUN pip install --no-cache-dir -r requirements.txt

# Copy the contents of the directory the Dockerfile is into the working directory of the to be container
COPY ./ $FLYWHEEL/

# Configure entrypoint
RUN bash -c 'chmod +rx $FLYWHEEL/run.py' && \
    bash -c 'chmod +rx $FLYWHEEL/app/' && \
    bash -c ' chmod +x /opt/ants-2.5.4/bin/antsRegistrationSyNQuick.sh'

ENTRYPOINT ["python", "/flywheel/v0/run.py"] 