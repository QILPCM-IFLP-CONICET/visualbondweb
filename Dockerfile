FROM python:3.11-slim

WORKDIR /app

# System deps for pymatgen/numpy, and git to install spectrojotometer
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ git \
    && rm -rf /var/lib/apt/lists/*

# Dependencies first, so this layer is cached while only the app changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml setup.py README.md ./
COPY visualbondweb ./visualbondweb
RUN pip install --no-cache-dir --no-deps .

EXPOSE 8000

# Same entry point as render.yaml
CMD ["uvicorn", "visualbondweb.api:app", "--host", "0.0.0.0", "--port", "8000"]
