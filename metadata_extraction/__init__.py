"""LoRA fine-tuning of a small LLM to extract JSON metadata from chunks.

Data flow: ``gold_paper_metadata`` -> ``grounded_training_examples`` ->
``extraction_prompt`` -> ``lora_finetuning_experiment`` (GPU) ->
``extraction_evaluation``, with ``paper_metadata_schema`` validating both the
training targets and the model's answers.
"""
