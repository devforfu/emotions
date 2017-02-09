FROM continuumio/miniconda3

WORKDIR /root
RUN mkdir data
COPY data data/
COPY bin/deploy.py deploy.py

RUN cd
RUN conda install numpy pandas matplotlib theano tensorflow h5py keras scikit-learn jupyter -y
RUN git clone https://github.com/devforfu/emotions emotions
RUN mv data/ emotions/data
RUN python deploy.py

ENV TINI_VERSION v0.6.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini

ENTRYPOINT ["/usr/bin/tini", "--"]

EXPOSE 8080
CMD jupyter notebook --port=8080 --no-browser --ip=0.0.0.0 --certfile=certs/jupyter.pem --keyfile=certs/jupyter.key
