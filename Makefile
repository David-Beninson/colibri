METAL ?= 0

.PHONY: all qwen36 metal-test qwen36-metal-test test check clean install uninstall

all qwen36 metal-test qwen36-metal-test test check clean install uninstall:
	$(MAKE) -C c $@ METAL=$(METAL)
