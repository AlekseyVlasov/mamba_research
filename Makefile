.PHONY: all_experiments

# Define parameter values
tokens_num_vals := 1 10 30 50 100
period_vals := 100 500 1000 2000

# Generate all combinations of targets
experiments := $(foreach tokens_num,$(tokens_num_vals), \
	$(foreach period,$(period_vals), \
		experiment_$(tokens_num)_$(period)))

# Main target: run all experiments
all_experiments: $(experiments)

# Template for individual experiment targets
$(experiments):
	@tokens_num=$(word 2,$(subst _, ,$@)) \
	 && period=$(word 3,$(subst _, ,$@)) \
	 && echo "Running experiment with tokens_num=$$tokens_num and period=$$period" \
	 && python src/train.py --config embeddings.yaml --tokens_num $$tokens_num --period $$period
