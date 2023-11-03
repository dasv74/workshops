#!/bin/bash

# This allow the build of the Jupyter Book and the publication on 
# github (gh-pages)
# The environment require 
# pip install -U jupyter-book
# pip install ghp-import

# Publish on: https://dasv74.github.io/workshops/

# From the repo directory directory (git repo - branch gh-pages)

jupyter-book build book/
ghp-import -n -p -f book/_build/html