#!/bin/bash

jupyter-book build book/
find . -name ".DS_Store" -print -delete

git status
git add .
git commit -m 'update'
git push



## book
## 


