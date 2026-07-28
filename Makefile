# OptiQuery
#
# Everything runs inside Docker, so the only host requirement is Docker Compose.
# `make seed` and `make baseline` execute in the `seeder` container, which is
# built from the repo Dockerfile and reaches the databases over the compose
# network rather than the published ports.

COMPOSE := docker compose
RUN     := $(COMPOSE) run --rm seeder

.DEFAULT_GOAL := help
.PHONY: help up down clean build seed baseline test test-live reset \
        optimize artifacts serve psql-primary psql-shadow logs

# Overridable on the command line: `make optimize QUERY="SELECT ..."` or
# `make optimize SEED=q2`.
QUERY ?=
SEED  ?= q2
OUT   ?= runs

help: ## Show available targets
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

up: ## Start the primary and shadow databases
	$(COMPOSE) up -d primary shadow

down: ## Stop the databases, keeping their data volumes
	$(COMPOSE) down

clean: ## Stop the databases and DELETE their data volumes
	$(COMPOSE) down -v

build: ## Build the application image used by the seeder and CLI
	$(COMPOSE) build seeder

seed: up build ## Load 4.75M rows into both databases, then VACUUM ANALYZE
	$(RUN) python seed/generate_data.py --target both

baseline: ## Time the four seed queries against primary and check they are slow
	$(RUN) python seed/measure_baseline.py

test: up ## Run the test suite against the seeded databases
	$(RUN) pytest -m "not live"

test-live: up ## Also run the tests that call a real model provider (costs tokens)
	$(RUN) pytest -m live -rs

optimize: up ## Optimise one query: make optimize SEED=q2, or QUERY="SELECT ..."
	@if [ -n "$(QUERY)" ]; then \
		$(RUN) python cli.py "$(QUERY)" --output $(OUT); \
	else \
		$(RUN) python cli.py --seed $(SEED) --output $(OUT); \
	fi

# Writes into the directory the Phase 6 frontend reads at build time, so the
# deployed site is a set of real measured runs rather than fixtures.
artifacts: up ## Run all four seed queries and write JSON for the frontend
	$(RUN) python cli.py --all --output frontend/public/runs

serve: up ## Serve POST /optimize on http://localhost:8000 (docs at /docs)
	$(COMPOSE) run --rm -p 8000:8000 seeder \
		uvicorn app.main:app --host 0.0.0.0 --port 8000

reset: clean seed ## Drop everything and reseed from scratch

psql-primary: ## Interactive psql against primary
	$(COMPOSE) exec primary psql -U optiquery -d optiquery

psql-shadow: ## Interactive psql against shadow
	$(COMPOSE) exec shadow psql -U optiquery -d optiquery

logs: ## Tail database logs
	$(COMPOSE) logs -f primary shadow
