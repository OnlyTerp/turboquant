FROM vllm/vllm-openai:nightly

# Install TurboQuant plugin
COPY src/ /opt/turboquant/src/
COPY vllm_plugin/ /opt/turboquant/vllm_plugin/
COPY setup.py /opt/turboquant/setup.py

RUN pip install -e /opt/turboquant/ --no-deps

# Set defaults
ENV VLLM_ATTENTION_BACKEND=turboquant
ENV HF_TOKEN=""

ENTRYPOINT ["python", "-m", "vllm.entrypoints.openai.api_server"]
