run:
	conda run --no-capture-output -n Error-Detection-at-Scale-mp python pipeline.py

install:
	conda create -n Error-Detection-at-Scale-mp python=3.10
	conda run --no-capture-output -n Error-Detection-at-Scale-mp pip install -r requirements.txt
	conda run --no-capture-output -n Error-Detection-at-Scale-mp python -m nltk.downloader stopwords

setup-santos:
	mkdir -p marshmallow_pipeline/santos/benchmark/
	mkdir -p marshmallow_pipeline/santos/stats/
	mkdir -p marshmallow_pipeline/santos/hashmap/
	mkdir -p marshmallow_pipeline/santos/groundtruth/
	mkdir -p marshmallow_pipeline/santos/yago/yago-original
	curl --remote-name-all https://yago-knowledge.org/data/yago4/full/2020-02-24/{yago-wd-annotated-facts.ntx.gz,yago-wd-class.nt.gz,yago-wd-facts.nt.gz,yago-wd-full-types.nt.gz,yago-wd-labels.nt.gz,yago-wd-sameAs.nt.gz,yago-wd-schema.nt.gz,yago-wd-shapes.nt.gz,yago-wd-simple-types.nt.gz} --output-dir marshmallow_pipeline/santos/yago/yago-original
	gzip -v -d marshmallow_pipeline/santos/yago/yago-original/*.gz
	mkdir marshmallow_pipeline/santos/yago/yago_pickle
	conda run --no-capture-output -n Error-Detection-at-Scale-mp python3 marshmallow_pipeline/santos/codes/preprocess_yago.py
	conda run --no-capture-output -n Error-Detection-at-Scale-mp python3 marshmallow_pipeline/santos/codes/Yago_type_counter.py
	conda run --no-capture-output -n Error-Detection-at-Scale-mp python3 marshmallow_pipeline/santos/codes/Yago_subclass_extractor.py
	conda run --no-capture-output -n Error-Detection-at-Scale-mp python3 marshmallow_pipeline/santos/codes/Yago_subclass_score.py

uninstall:
	conda remove -n Error-Detection-at-Scale-mp --all

.PHONY: run, install, uninstall, setup-santos
.DEFAULT_GOAL := run
