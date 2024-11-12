JL = julia 

temperature_type ?= SigmoidGradient
n ?= 30
m ?= 70
nbatch ?= 100000
width ?= 1.0
m_step ?= 9
m_minimum ?= 18
m_maximum ?= 300

v_maximum ?= 0.08
example_num = 100
v_minimum ?= 0.00001
device ?= 1

calculate_datato50:
	$(JL) datato50.jl $(temperature_type) $(n) $(nbatch) $(width) $(m_step) $(m_minimum) $(m_maximum) $(device)

calculate_error_vs_sweep:
	$(JL) data_error_vs_sweep.jl $(temperature_type) $(nbatch) $(n) $(m) $(width) $(example_num) $(v_maximum) $(v_minimum) $(device)

# .PHONY: calculate_datato50
# .PHONY: calculate_error_vs_sweep
