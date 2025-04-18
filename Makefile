NAME?=sensorium
COMMAND?=bash
OPTIONS?=

GPUS?=all
ifeq ($(GPUS),none)
	GPUS_OPTION=
else
	GPUS_OPTION=--gpus=$(GPUS)
endif

.PHONY: all
all: stop build run

.PHONY: build
build:
	docker build -t $(NAME) .

.PHONY: stop
stop:
	-docker stop $(NAME)
	-docker rm $(NAME)

.PHONY: run
run:
	docker run --rm -dit \
		--net=host \
		--ipc=host \
		$(OPTIONS) \
		$(GPUS_OPTION) \
		-v $(shell pwd):/workdir \
		--name=$(NAME) \
		$(NAME) \
		$(COMMAND)
	docker attach $(NAME)

.PHONY: attach
attach:
	docker attach $(NAME)

.PHONY: logs
logs:
	docker logs -f $(NAME)

.PHONY: exec
exec:
	docker exec -it $(OPTIONS) $(NAME) $(COMMAND)
