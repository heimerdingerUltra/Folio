# Image of the command-line tool, built in CI from the wheel in dist/.
#
#   docker run --rm -v "$PWD:/work" ghcr.io/heimerdingerultra/folio --input data --output results.json
#
# Base install only: tables use the heuristic detector, and pages without a
# text layer are reported, not read (no layout model, no OCR in the image).
FROM python:3.12-slim

LABEL org.opencontainers.image.source="https://github.com/heimerdingerUltra/Folio" \
      org.opencontainers.image.description="Parse PDFs into RAG-ready chunks"

COPY dist/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/*.whl && rm /tmp/*.whl

WORKDIR /work
ENTRYPOINT ["process-pdfs"]
CMD ["--help"]
