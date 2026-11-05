ifneq ($(wildcard /Library/Developer/CommandLineTools),)
export DEVELOPER_DIR ?= /Library/Developer/CommandLineTools
export PATH := /Library/Developer/CommandLineTools/usr/bin:$(PATH)
endif
METAL ?= 1

.PHONY: all qwen36 metal-test qwen36-metal-test test check clean install uninstall

all qwen36 metal-test qwen36-metal-test test check clean install uninstall:
	$(MAKE) -C c $@ METAL=$(METAL)


