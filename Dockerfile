FROM nvcr.io/nvidia/tensorflow:23.08-tf2-py3

RUN pip install scikit-learn --upgrade
RUN pip install dask
RUN pip install dask-jobqueue
RUN pip install statsmodels
RUN pip install lightgbm
RUN pip install xgboost
RUN pip install matplotlib
RUN pip install seaborn
RUN pip uninstall --yes numpy
RUN pip install "numpy <=1.24.3, >=1.22.3"
RUN pip install torch torchvision pyod lightning




