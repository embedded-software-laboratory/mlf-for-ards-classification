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
    torch torchvision pyod pytorch-lightning \
    timm \
    albumentations \
    torchmetrics \
    tqdm \
    opencv-python-headless \
    libauc

RUN pip uninstall --yes numpy
RUN pip install "numpy <=1.24.3, >=1.22.3"
