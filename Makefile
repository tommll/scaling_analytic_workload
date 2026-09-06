COMPOSE  := docker compose -f docker/docker-compose.yml
PROFILES := --profile w1 --profile w2 --profile w3 --profile w4 --profile w5 --profile w6
SCALE    ?= m
NODES    ?= 4

.PHONY: help build up down ps gen bench bench-quick calibrate parity plot demo-fault demo-skew plan shell clean distclean

help:
	@grep -hE '^[a-zA-Z-]+:.*?## ' $(MAKEFILE_LIST) | awk -F':.*?## ' '{printf "  \033[1m%-14s\033[0m %s\n", $$1, $$2}'

build: ## Build the single image used by every role
	$(COMPOSE) $(PROFILES) build

up: ## Create the cluster (master, 6 pinned worker nodes, driver, baselines)
	$(COMPOSE) $(PROFILES) up -d
	@echo "Spark master UI -> http://localhost:8080"

down: ## Tear everything down
	$(COMPOSE) $(PROFILES) down --remove-orphans

ps: ## Show which nodes are up
	@$(COMPOSE) $(PROFILES) ps --format '{{.Name}}\t{{.State}}'

gen: ## Generate the dataset (SCALE=xs|s|m|l|xl)
	docker exec baseline-big python3 -m src.gen.generate --scale $(SCALE)

EXPERIMENTS ?= calibrate,strong,small,memory
SKIP_TIERS  ?=

bench: ## Full suite -> results/results.json (~40 min). EXPERIMENTS=slots for the slot sweep.
	python3 -m src.bench.runner --experiments $(EXPERIMENTS) \
	  --scale $(SCALE) --memory-scale l --small-scale xs --nodes 1,2,4 \
	  --slots 1,2,4,10 --reps 3 --skip-tiers "$(SKIP_TIERS)"

bench-quick: ## Fast sanity run of the suite (~6 min)
	python3 -m src.bench.runner --experiments strong --scale s --nodes 1,2 --reps 1

calibrate: ## Measure the HOST's parallel ceiling. Run this before trusting any curve.
	python3 -m src.bench.runner --experiments calibrate --out results/calibration.json

parity: ## Assert all three tiers produce the same answer
	docker exec baseline python3 -m src.bench.parity /data/out pandas chunked spark

plot: ## Charts + RESULTS.md from results/results.json
	docker exec baseline-big python3 -m src.bench.plot /results/results.json /results

demo-fault: ## Kill a node mid-job; prove the answer is unchanged
	python3 -m src.bench.fault_demo

demo-skew: ## Hot-key skew on a shuffle join, with and without salting
	@# Two regimes: skew only hurts once one key holds more than 1/cores of the
	@# work, so 30% on a 4-core cluster is nearly harmless and 85% is not.
	@for share in 0.30 0.85; do \
	  docker exec baseline-big python3 -m src.gen.generate \
	    --scale $(SCALE) --skew --hot-share $$share; \
	  docker exec -e SPARK_CORES_MAX=$(NODES) -e SPARK_EXPECT_EXECUTORS=$(NODES) \
	    spark-client python3 -m src.distributed.skew_demo --nodes $(NODES) \
	    --per-zone 200 --out /results/skew_$$share.json 2>/dev/null \
	    | grep -E "hottest|cores ->|predicted|plain|salted|salting"; \
	done
	@echo "note: regenerate unskewed data with 'make gen' before re-running benchmarks"

plan: ## Dump the Spark physical plan (proof of broadcast join + pushdown)
	docker exec -e SPARK_CORES_MAX=$(NODES) -e SPARK_EXPECT_EXECUTORS=$(NODES) \
	  spark-client python3 -m src.distributed.spark_pipeline \
	  --out /data/out/spark --dump-plan /results/spark_plan.txt
	@grep -m1 -o 'BroadcastHashJoin' /results/spark_plan.txt 2>/dev/null \
	  || grep -m1 -o 'BroadcastHashJoin' results/spark_plan.txt

shell: ## Shell into the driver
	docker exec -it spark-client bash

clean: ## Remove generated data and outputs
	rm -rf data/trips data/zones data/out

distclean: down clean ## Everything, including results
	rm -rf results/*.png results/*.json results/*.txt results/RESULTS.md results/*.log
