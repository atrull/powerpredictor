buildconfig:
	uv pip install pyinstaller

build: dist/power_predictor
	uv run pyinstaller --onefile power_predictor.py

build-nuitka:
	mkdir -p dist && uv run nuitka --standalone --onefile --enable-plugin=anti-bloat --include-package=matplotlib --output-dir=dist power_predictor.py

build-pyinstaller-opt: power_predictor.spec
	uv run pyinstaller power_predictor.spec

dist/power_predictor:
	uv run pyinstaller --onefile power_predictor.py

build-test: build
	./dist/power_predictor k20_pull.csv --weight 1060 --occupant 130 --displacement 2.0 --out k20.png

k20:
	MPLBACKEND=Agg uv run python power_predictor.py k20_pull.csv --weight 1060 --occupant 130 --displacement 2.0 --out k20.png
	MPLBACKEND=Agg uv run python power_predictor.py k20_pull2.csv --weight 1060 --occupant 130 --displacement 2.0 --out k20_pull2.png

k20-unfiltered:
	MPLBACKEND=Agg uv run python power_predictor.py k20_pull.csv --weight 1060 --occupant 130 --displacement 2.0 --no-rpm-filtering --out k20-unfiltered.png
	MPLBACKEND=Agg uv run python power_predictor.py k20_pull2.csv --weight 1060 --occupant 130 --displacement 2.0 --no-rpm-filtering --out k20_pull2-unfiltered.png

white:
	MPLBACKEND=Agg uv run python power_predictor.py white_nbg_run.csv --weight 1060 --occupant 130 --final-drive 4.78 --displacement 1.6 --max-gap 10 --out white_nbg_run.png
	MPLBACKEND=Agg uv run python power_predictor.py white_pull2.csv --weight 1060 --occupant 130 --gear-ratio 1.33 --final-drive 4.78 --displacement 1.6 --max-gap 10 --out white_pull2.png
	MPLBACKEND=Agg uv run python power_predictor.py white_2_runs.csv --weight 1060 --occupant 130 --gear-ratio 1.0 --final-drive 4.78 --displacement 1.6 --max-gap 10 --out white_2_runs.png
	MPLBACKEND=Agg uv run python power_predictor.py white_2_runs2.csv --weight 1060 --occupant 130 --gear-ratio 1.0 --final-drive 4.78 --displacement 1.6 --max-gap 10 --out white_2_runs2.png

white-downsampled:
	MPLBACKEND=Agg uv run python power_predictor.py white_nbg_run.csv --downsample-hz 15 --weight 1060 --occupant 130 --final-drive 4.78 --displacement 1.6 --max-gap 10 --out white_nbg_run.png
	MPLBACKEND=Agg uv run python power_predictor.py white_pull2.csv --downsample-hz 15 --weight 1060 --occupant 130 --gear-ratio 1.33 --final-drive 4.78 --displacement 1.6 --max-gap 10 --out white_pull2.png
	MPLBACKEND=Agg uv run python power_predictor.py white_2_runs.csv --downsample-hz 15 --weight 1060 --occupant 130 --gear-ratio 1.0 --final-drive 4.78 --displacement 1.6 --max-gap 10 --out white_2_runs.png
	MPLBACKEND=Agg uv run python power_predictor.py white_2_runs2.csv --downsample-hz 15 --weight 1060 --occupant 130 --gear-ratio 1.0 --final-drive 4.78 --displacement 1.6 --max-gap 10 --out white_2_runs2.png

white-unfiltered:
	MPLBACKEND=Agg uv run python power_predictor.py white_nbg_run.csv --weight 1060 --occupant 130 --final-drive 4.78 --displacement 1.6 --max-gap 10 --out white_nbg_run.png --no-rpm-filtering
	MPLBACKEND=Agg uv run python power_predictor.py white_pull2.csv --weight 1060 --occupant 130 --gear-ratio 1.33 --final-drive 4.78 --displacement 1.6 --max-gap 10 --out white_pull2.png --no-rpm-filtering
	MPLBACKEND=Agg uv run python power_predictor.py white_2_runs.csv --weight 1060 --occupant 130 --gear-ratio 1.0 --final-drive 4.78 --displacement 1.6 --max-gap 10 --out white_2_runs.png --no-rpm-filtering
	MPLBACKEND=Agg uv run python power_predictor.py white_2_runs2.csv --weight 1060 --occupant 130 --gear-ratio 1.0 --final-drive 4.78 --displacement 1.6 --max-gap 10 --out white_2_runs2.png --no-rpm-filtering

black:
	# should be 4th gear 1.0 ratio but that predicts too much power
	MPLBACKEND=Agg uv run python power_predictor.py black.csv --weight 1022 --occupant 85 --gear-ratio 1.33 --final-drive 4.3 --displacement 1.6 --smoothing-factor 5 --max-gap 15 --out black.png
	MPLBACKEND=Agg uv run python power_predictor.py black.csv --weight 1022 --occupant 85 --gear-ratio 1.33 --final-drive 4.3 --displacement 1.6 --smoothing-factor 10 --max-gap 15 --out black-smoother.png

all: k20 white black
