# Authors: Brian Chang, Farhan Faisal, Daria Khon, Jay Mangat
# Date: 2024-12-31

quarto: analysis/report.qmd
	quarto render analysis/report.qmd --to html
	quarto render analysis/report.qmd --to pdf
	cp analysis/report.html docs/
	mv docs/report.html docs/index.html

.PHONY: \
	all quarto \
	clean-data clean-tables clean-figures clean-models clean-markers clean