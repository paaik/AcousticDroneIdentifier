# Project setup
PROJ = tdm_aggregator_top
TOP  = tdm_aggregator_top

# Sources
FILES = \
	i2s_mic_timing_gen.v \
	mic_capture_4lane.v \
	pi_tdm_serializer.v \
	superframe_loader.v \
	tdm_aggregator_top.v

# Constraints / device
PCF     = constraints/tdm_aggregator_top.pcf
DEVICE  = hx8k
PACKAGE = cb132

# Programmer
PROGRAMMER = ./iceFUNprog/iceFUNprog
PORT ?=

.PHONY: all iceFUN burn programmer clean

all: iceFUN

iceFUN:
	# Synthesize using Yosys
	yosys -p "synth_ice40 -top $(TOP) -json $(PROJ).json" $(FILES)
	# Place and route using nextpnr
	nextpnr-ice40 --$(DEVICE) --package $(PACKAGE) --json $(PROJ).json --pcf $(PCF) --asc $(PROJ).asc --opt-timing
	# Convert to bitstream using icepack
	icepack $(PROJ).asc $(PROJ).bin

programmer:
	$(MAKE) -C iceFUNprog

burn: iceFUN programmer
	$(PROGRAMMER) $(if $(PORT),-P $(PORT),) $(PROJ).bin

clean:
	rm -f $(PROJ).json $(PROJ).asc $(PROJ).bin
