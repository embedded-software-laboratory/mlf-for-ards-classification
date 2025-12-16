FROM nvcr.io/nvidia/tensorflow:23.08-tf2-py3

RUN pip install scikit-learn --upgrade
RUN pip install pandas
RUN pip install pydantic
RUN pip install pillow
RUN pip install pyyaml
RUN pip install scikit-image
RUN pip install statsmodels
RUN pip install lightgbm
RUN pip install xgboost
RUN pip install matplotlib
RUN pip install seaborn
RUN pip uninstall --yes numpy
RUN pip install "numpy <=1.24.3, >=1.22.3"
RUN pip install torch torchvision pyod lightning
RUN pip install albumentations
RUN pip install torchmetrics
RUN pip install tqdm
RUN pip install opencv-python-headless

# Clean pip cache to reduce image size
RUN pip cache purge || true

