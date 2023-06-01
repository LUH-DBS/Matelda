run:
	conda run --no-capture-output -n Error-Detection-at-Scale python3 pipeline.py

install:
	conda create -n Error-Detection-at-Scale python=3.9
	conda run --no-capture-output -n Error-Detection-at-Scale pip install -r requirements.txt

setup-santos:
	cd marshmallow_pipeline/santos
	mkdir benchmark && cd benchmark
	zenodo_get 7758091
	unzip santos_benchmark
	cd santos_benchmark
	rm *.csv
	cd ..
	unzip real_data_lake_benchmark
	cd real_data_lake_benchmark
	rm *.csv
	cd ..
	mv real_data_lake_benchmark real_tables_benchmark
	rm *.zip
	cd ../../..
	mkdir -p marshmallow_pipeline/santos/yago/yago-original
	curl --remote-name-all https://yago-knowledge.org/data/yago4/full/2020-02-24/{yago-wd-class.nt.gz,yago-wd-facts.nt.gz,yago-wd-full-types.nt.gz,yago-wd-labels.nt.gz,yago-wd-schema.nt.gz,yago-wd-simple-types.nt.gz,yago-wd-schema.nt.gz} --output-dir marshmallow_pipeline/santos/yago/yago-original
	gzip -v -d marshmallow_pipeline/santos/yago/yago-original/*.gz
	cd marshmallow_pipeline/santos/codes
	conda run --no-capture-output -n Error-Detection-at-Scale python3 preprocess_yago.py
	conda run --no-capture-output -n Error-Detection-at-Scale python3 Yago_type_counter.py
	conda run --no-capture-output -n Error-Detection-at-Scale python3 Yago_subclass_extractor.py
	conda run --no-capture-output -n Error-Detection-at-Scale python3 Yago_subclass_score.py
	cd ../../../

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

.PHONY: run, install, uninstall, setup-santos, clean-santos, clean-logs
.DEFAULT_GOAL := run