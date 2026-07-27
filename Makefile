# OptiQuery
#
# Everything runs inside Docker, so the only host requirement is Docker Compose.
# `make seed` and `make baseline` execute in the `seeder` container, which is
# built from the repo Dockerfile and reaches the databases over the compose
# network rather than the published ports.

COMPOSE := docker compose
RUN     := $(COMPOSE) run --rm seeder

.DEFAULT_GOAL := help
.PHONY: help up down clean build seed baseline test reset psql-primary psql-shadow logs

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
	$(RUN) pytest

reset: clean seed ## Drop everything and reseed from scratch

psql-primary: ## Interactive psql against primary
	$(COMPOSE) exec primary psql -U optiquery -d optiquery

psql-shadow: ## Interactive psql against shadow
	$(COMPOSE) exec shadow psql -U optiquery -d optiquery

logs: ## Tail database logs
	$(COMPOSE) logs -f primary shadow
