run:
	conda run --no-capture-output -n Error-Detection-at-Scale python3 pipeline.py

install:
	conda create -n Error-Detection-at-Scale python=3.9
	conda run --no-capture-output -n Error-Detection-at-Scale pip install -r requirements.txt

setup-santos:
	conda run --no-capture-output -n Error-Detection-at-Scale zenodo_get 7758091 -o marshmallow_pipeline/santos/benchmark
	unzip marshmallow_pipeline/santos/benchmark/santos_benchmark.zip -d marshmallow_pipeline/santos/benchmark
	rm marshmallow_pipeline/santos/benchmark/santos_benchmark/*.csv
	unzip marshmallow_pipeline/santos/benchmark/real_data_lake_benchmark.zip -d marshmallow_pipeline/santos/benchmark
	rm marshmallow_pipeline/santos/benchmark/real_data_lake_benchmark/*.csv
	mv marshmallow_pipeline/santos/benchmark/real_data_lake_benchmark marshmallow_pipeline/santos/benchmark/real_tables_benchmark
	rm marshmallow_pipeline/santos/benchmark/*.zip
	mkdir -p marshmallow_pipeline/santos/benchmark/
	mkdir -p marshmallow_pipeline/santos/yago/yago-original
	curl --remote-name-all https://yago-knowledge.org/data/yago4/full/2020-02-24/{yago-wd-class.nt.gz,yago-wd-facts.nt.gz,yago-wd-full-types.nt.gz,yago-wd-labels.nt.gz,yago-wd-schema.nt.gz,yago-wd-simple-types.nt.gz,yago-wd-schema.nt.gz} --output-dir marshmallow_pipeline/santos/yago/yago-original
	gzip -v -d marshmallow_pipeline/santos/yago/yago-original/*.gz
	mkdir marshmallow_pipeline/santos/yago/yago_pickle
	conda run --no-capture-output -n Error-Detection-at-Scale python3 marshmallow_pipeline/santos/codes/preprocess_yago.py
	conda run --no-capture-output -n Error-Detection-at-Scale python3 marshmallow_pipeline/santos/codes/Yago_type_counter.py
	conda run --no-capture-output -n Error-Detection-at-Scale python3 marshmallow_pipeline/santos/codes/Yago_subclass_extractor.py
	conda run --no-capture-output -n Error-Detection-at-Scale python3 marshmallow_pipeline/santos/codes/Yago_subclass_score.py

clean-santos:
	rm -rf marshmallow_pipeline/santos/benchmark/*
	rm -rf marshmallow_pipeline/santos/stats/*
	rm -rf marshmallow_pipeline/santos/hashmap/*
	rm -rf marshmallow_pipeline/santos/groundtruth/*
	rm -rf results

clean-yago:
	rm -rf marshmallow_pipeline/santos/yago/*

clean-logs:
	rm -rf marshmallow_pipeline/logs
	rm -rf logs

clean-all: clean-santos clean-yago clean-logs

uninstall:
	conda remove -n Error-Detection-at-Scale --all

.PHONY: run, install, uninstall, setup-santos, clean-santos, clean-logs, clean-all, clean-yago
.DEFAULT_GOAL := run