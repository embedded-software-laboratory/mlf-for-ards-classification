FROM nvcr.io/nvidia/tensorflow:23.08-tf2-py3

# 🔒 NumPy 1.x erzwingen (TF + pandas kompatibel)
RUN pip install --no-cache-dir "numpy<2"

# Alles andere danach
RUN pip install --no-cache-dir \
    pandas \
    scikit-learn \
    statsmodels \
    lightgbm \
    xgboost \
    matplotlib \
    seaborn \
    pydantic \
    pillow \
    pyyaml \
    scikit-image \
    torch torchvision pyod lightning \
    albumentations \
    torchmetrics \
    tqdm \
    opencv-python-headless
