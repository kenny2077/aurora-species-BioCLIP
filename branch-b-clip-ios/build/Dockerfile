# Dockerfile — isolated, pinned environment for the BioCLIP-2 -> Core ML conversion.
# coremltools is version-sensitive: torch 2.7.0 + torchvision 0.22.0 is the
# tested pair (9.x converts both; newer torch broke the frontend twice).
#
#   docker build -t aurora-coreml-build build/
#   # pre-seed a host HF cache to skip the 1.7 GB weight download:
#   docker run --rm -v "$PWD:/pkg" -v "$HOME/.cache/huggingface:/hf-cache" \
#     -e HF_HOME=/hf-cache aurora-coreml-build
#   # otherwise it downloads into the image:
#   docker run --rm -v "$PWD:/pkg" aurora-coreml-build
FROM python:3.10-slim

RUN pip install --no-cache-dir \
    torch==2.7.0 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir torchvision==0.22.0 \
    open_clip_torch "coremltools>=9.0" numpy pillow

WORKDIR /pkg
CMD ["python", "convert_coreml.py"]
