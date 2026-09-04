# VIGÍA — the commands you actually need.
.PHONY: dev dev-loop dev-offline demo check test figures paper arxiv docs clips

## dev: start the whole demonstration and open the browser
##      Recorded clips play once; the public live cameras keep updating.
dev:
	@.venv/bin/python scripts/demo.py

## dev-loop: the same, with the recorded clips replaying continuously
dev-loop:
	@.venv/bin/python scripts/demo.py --loop

## dev-offline: clips only, no public live cameras
dev-offline:
	@.venv/bin/python scripts/demo.py --no-live

demo: dev

## check: the three guards plus the test suite
check:
	@.venv/bin/python tools/check_licence.py
	@.venv/bin/python tools/check_privacy.py
	@.venv/bin/python tools/freeze_results.py
	@for t in tests/*.py; do .venv/bin/python $$t || exit 1; done

test: check

## clips: rebuild the demonstration clips from local data
clips:
	@.venv/bin/python scripts/make_demo_clips.py

## figures: regenerate the charts from the stored results
figures:
	@.venv/bin/python docs/figures.py

## paper: rebuild the technical paper as PDF (LaTeX; regenerates figures
##        and the numbers macros from eval/results first)
paper:
	@.venv/bin/python docs/build_paper.py

## arxiv: package the paper for arXiv (dist/vigia-arxiv.tar.gz)
arxiv:
	@.venv/bin/python docs/make_arxiv.py

## docs: the superseded Word build, kept for reference
docs:
	@.venv/bin/python docs/build_documentation.py
