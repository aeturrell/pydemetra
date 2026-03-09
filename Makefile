# This makes the documentation and readme for pydemetra

.PHONY: all clean site publish

all: README.md site

# Build the readme
README.md: docs/index.qmd
		cp docs/index.qmd README.md


# Build the github pages site
site:
		uv pip install -e .
		uv run quartodoc build --config docs/_quarto.yml
		cd docs; uv run quarto render --execute
		rm docs/.gitignore
		uv run nbstripout docs/*.ipynb
		uv run pre-commit run --all-files


clean:
	rm README.md


publish:
		uv pip install -e .
		uv run quartodoc build --config docs/_quarto.yml
		cd docs;uv run quarto render --execute
		cd docs;uv run quarto publish gh-pages --no-render
		rm docs/.gitignore
		uv run nbstripout docs/*.ipynb
		uv run pre-commit run --all-files
