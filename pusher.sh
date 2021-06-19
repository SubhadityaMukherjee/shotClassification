black "." && isort .
jupytext --to py main.ipynb
jupytext --to html main.ipynb
if [[ ! -z $1 ]]; then
        git add . && git commit -m $1 && git push
fi
