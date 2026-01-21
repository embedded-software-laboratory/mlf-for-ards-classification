FROM nvcr.io/nvidia/tensorflow:23.08-tf2-py3

RUN apt-get update && \
    apt-get install -y libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Alles andere danach
RUN pip install --no-cache-dir \
    pandas \
    scikit-learn \
    scipy \
    sympy \
    joblib \
    statsmodels \
    lightgbm \
    xgboost \
    matplotlib \
    seaborn \
    pydantic \
    pillow \
    pyyaml \
    scikit-image \
    pyod \
    pytorch-lightning \
    timm \
    albumentations \
    torchmetrics \
    tqdm \
    opencv-python-headless \
    libauc

# Install PyTorch with CUDA 12.1 support (compatible with CUDA 12.8 driver)
RUN pip install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

RUN pip uninstall --yes numpy
RUN pip install "numpy <=1.24.3, >=1.22.3"
