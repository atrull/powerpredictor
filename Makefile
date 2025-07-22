all: k20 k20-unfiltered white white-unfiltered

k20:
	MPLBACKEND=Agg uv run python power_predictor.py k20_pull.csv --weight 1060 --occupant 130 --displacement 2.0 --out k20.png 

k20-unfiltered:
	MPLBACKEND=Agg uv run python power_predictor.py k20_pull.csv --weight 1060 --occupant 130 --displacement 2.0 --no-rpm-filtering --out k20-unfiltered.png

white:
	MPLBACKEND=Agg uv run python power_predictor.py white_nbg_run.csv --weight 1060 --occupant 130 --displacement 1.6 --max-gap 10 --out white_nbg_run.png 

white-unfiltered:
	MPLBACKEND=Agg uv run python power_predictor.py white_nbg_run.csv --weight 1060 --occupant 130 --displacement 1.6 --max-gap 10 --no-rpm-filtering --out white_nbg_run-unfiltered.png