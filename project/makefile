JL = julia 

temperature_type ?= SigmoidGradient
n ?= 16
m ?= 70
nbatch ?= 10000
width ?= 1.0
m_step ?= 3
m_minimum ?= 6
m_maximum ?= 100
input_fixed ?= true

v_maximum ?= 0.012
example_num = 30
v_minimum ?= 0.00001
device ?= 4

calculate_datato50:
	$(JL) datato50.jl $(temperature_type) $(n) $(nbatch) $(width) $(m_step) $(m_minimum) $(m_maximum) $(input_fixed) $(device)

calculate_error_vs_sweep:
	$(JL) data_error_vs_sweep.jl $(temperature_type) $(nbatch) $(n) $(m) $(width) $(example_num) $(v_maximum) $(v_minimum) $(device)

calculate_datato50_spin:
	$(JL) datato50_spin.jl $(temperature_type) $(n) $(nbatch) $(width) $(m_step) $(m_minimum) $(m_maximum) $(device)

# .PHONY: calculate_datato50
# .PHONY: calculate_error_vs_sweep
