#/bin/bash

python tests/testLike.py;
python -m doctest -v actExamples/likelihood.py;
