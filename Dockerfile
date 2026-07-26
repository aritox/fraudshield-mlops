FROM python:3.12-slim-bookworm AS builder

WORKDIR /build
COPY pyproject.toml README.md ./
COPY artifacts/docker_build_wheels/ /build-wheels/
RUN python -m pip install --no-cache-dir --no-index --find-links=/build-wheels \
      packaging==26.2 setuptools==83.0.0 wheel==0.47.0 \
    && python -c "import importlib.metadata as m, pathlib, tomllib; from packaging.requirements import Requirement; requirements=[Requirement(value) for value in tomllib.loads(pathlib.Path('pyproject.toml').read_text(encoding='utf-8'))['build-system']['requires']]; assert all(requirement.specifier.contains(m.version(requirement.name), prereleases=True) for requirement in requirements)"
COPY src/ ./src/
RUN python -m pip wheel --no-cache-dir --no-deps --no-build-isolation --wheel-dir /wheels .

FROM python:3.12-slim-bookworm AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    FRAUDSHIELD_REPOSITORY_ROOT=/opt/fraudshield \
    FRAUDSHIELD_MODEL_URI=/opt/fraudshield/model/production_sgd \
    FRAUDSHIELD_MODEL_PACKAGE_MANIFEST=/opt/fraudshield/artifacts/container/model_package_manifest.json \
    MLFLOW_TRACKING_URI=file:///tmp/mlruns

WORKDIR /opt/fraudshield
COPY --from=builder /wheels /tmp/wheels
RUN python -m pip install --no-cache-dir \
      numpy==2.5.1 pandas==2.3.3 scikit-learn==1.9.0 joblib==1.5.3 \
      cloudpickle==3.1.2 PyYAML==6.0.3 pydantic==2.13.4 \
      fastapi==0.140.0 uvicorn==0.51.0 mlflow-skinny==3.14.0 \
      SQLAlchemy==2.0.51 "psycopg[binary]==3.3.4" alembic==1.18.5 \
    && python -m pip install --no-cache-dir --no-deps /tmp/wheels/*.whl \
    && rm -rf /tmp/wheels \
    && groupadd --gid 10001 fraudshield \
    && useradd --uid 10001 --gid 10001 --no-create-home --shell /usr/sbin/nologin fraudshield

COPY --chown=10001:10001 configs/api.yaml configs/database.yaml configs/mlflow.yaml ./configs/
COPY --chown=10001:10001 alembic.ini ./
COPY --chown=10001:10001 alembic/ ./alembic/
COPY --chown=10001:10001 artifacts/container_model/production_sgd/ ./model/production_sgd/
COPY --chown=10001:10001 artifacts/container/model_package_manifest.json ./artifacts/container/model_package_manifest.json

USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --start-period=20s --retries=6 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/ready', timeout=2).read()"]
CMD ["python", "-m", "uvicorn", "fraudshield.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
