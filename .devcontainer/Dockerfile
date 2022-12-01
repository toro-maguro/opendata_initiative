FROM python:3.10.4-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    apt-utils \
    gcc \
    build-essential \
    && pip install --no-cache-dir \
    autopep8 \
    flake8 \
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
RUN pip install --upgrade setuptools

# libraries
# RUN apt-get install -y vim less mecab libmecab-dev mecab-ipadic-utf8
# RUN cp /etc/mecabrc /usr/local/etc/mecabrc
RUN python -m pip install jupyterlab
RUN python -m pip install pandas
RUN python -m pip install numpy
RUN python -m pip install scipy
RUN python -m pip install scikit-learn
RUN python -m pip install lightgbm
RUN python -m pip install xgboost
RUN python -m pip install matplotlib
RUN python -m pip install seaborn
RUN python -m pip install WordCloud
RUN python -m pip install beautifulsoup4
RUN python -m pip install janome
RUN python -m pip install nlplot
RUN python -m pip install mca
RUN python -m pip install japanize-matplotlib