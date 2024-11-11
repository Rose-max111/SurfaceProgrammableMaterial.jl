JL = julia 

temperature_type ?= SigmoidGradient
n ?= 16
nbatch ?= 10000
width ?= 0.06
m_step ?= 9
m_minimum ?= 18
m_maximum ?= 300
device ?= 1

calculate:
	$(JL) datato50.jl $(temperature_type) $(n) $(nbatch) $(width) $(m_step) $(m_minimum) $(m_maximum) $(device)

.PHONY: calculate
