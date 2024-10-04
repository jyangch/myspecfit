FROM ubuntu2004-py38

WORKDIR /usr/src/BaySpec

COPY . .

ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "App.py", "--server.port=8501", "--server.address=0.0.0.0"]
