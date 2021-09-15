.PHONY: create_pipeline
create_pipeline:
	tfx pipeline create --pipeline_path fraud/local_runner.py

.PHONY: update_pipeline
update_pipeline:
	tfx pipeline update --pipeline_path fraud/local_runner.py

.PHONY: run_pipeline
run_pipeline:
	tfx run create --pipeline_name fraud
