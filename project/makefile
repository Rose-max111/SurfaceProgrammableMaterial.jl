JL = julia --project=..

device ?= 0
width ?= 12
depth ?= 10
gradient ?= 1.5
gauss_width ?= 1.0

calculate:
	$(JL) toymodel_genearalannealing.jl $(width) $(depth) $(gauss_width) $(gradient) $(device)

.PHONY: calculate
