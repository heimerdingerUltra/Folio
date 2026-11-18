"""PDF -> clean text -> RAG-ready chunks.

Data flow: ``pdf_layout_extraction`` -> ``chunking_strategies`` ->
``chunk_record_schema`` (validation) -> ``chunk_dataset_export``, orchestrated
by ``chunking_pipeline`` and measured by ``chunk_quality_metrics`` and
``retrieval_benchmark``.
"""

__version__ = "1.1.0"
