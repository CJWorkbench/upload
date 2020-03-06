# We run everything in a Dockerfile so we can pull arrow-tools binaries
FROM workbenchdata/arrow-tools:v0.0.11 as arrow-tools

FROM python:3.8.1-buster AS test

COPY --from=arrow-tools /usr/bin/csv-to-arrow /usr/bin/csv-to-arrow
COPY --from=arrow-tools /usr/bin/json-to-arrow /usr/bin/json-to-arrow
COPY --from=arrow-tools /usr/bin/xls-to-arrow /usr/bin/xls-to-arrow
COPY --from=arrow-tools /usr/bin/xlsx-to-arrow /usr/bin/xlsx-to-arrow

RUN pip install black pyflakes isort pytest

# README is read by setup.py
COPY setup.py README.md /app/
# __version__ (for setup.py)
COPY upload.py /app/upload.py
COPY test_upload.py /app/test_upload.py
COPY upload.yaml /app/upload.yaml
COPY locale /app/locale
WORKDIR /app
RUN pip install .

COPY . /app/

RUN true \
      && pyflakes . \
      && black --check . \
      && pytest --verbose
