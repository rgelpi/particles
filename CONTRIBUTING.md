Thanks a lot for contributing to particles!

Please don't send PR (pull requests) directly to the master branch; send them
instead:

* to the experimental branch if your PR is a bug fix; 

* to a new branch if your PR adds functionality. 

In fact, when you fork any git repository, it is good practice to switch to
another branch than master before making any change. In this way, your PRs will
be automatically sent to that branch. 

In doubt, contact the maintainer: nicolas dot chopin at ensae dot fr

For type checking, install the development extra with
`python -m pip install -e '.[typing]'`, then run `python -m pyright` in that
environment. The configuration in `pyproject.toml` checks the library in standard
mode against the minimum supported Python version. Existing diagnostics are
being addressed incrementally; keep remaining errors visible when making fixes.

Run the numerical regression checks with
`python -m unittest discover -s tests -p 'test_*.py'`.
