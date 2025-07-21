all: test test-unfiltered

test:
	MPLBACKEND=Agg uv run python power_predictor.py k20_pull.csv --weight 1060 --occupant 130 --displacement 2.0 --out test.png 

test-unfiltered:
	MPLBACKEND=Agg uv run python power_predictor.py k20_pull.csv --weight 1060 --occupant 130 --displacement 2.0 --no-rpm-filtering --out test-unfiltered.png