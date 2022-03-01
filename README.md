# README for model

The following repository contains code for the model used in my dissertation.
Specifically, this repository contains code to:
* Generate simulations of an individual agent's behavior over the course of their lifetime
* Generate dynamic simulations of how individual agent's behavior spills over into other cohorts

## Set-up

I use PyCharm Professional as my IDE for this project, hence the .idea folder.
It may be easiest to use Pycharm if cloning this repo.

I maintain helpful plotting tools in my [img_tools](https://github.com/tara-sullivan/img_tools) repository.
If you want to run any of programs that generate plots, this repo should also be cloned, and the path to this directory should be added to your python path.
This can be done in a pycharm project as follows (instructions adapted from [here](https://stackoverflow.com/a/48948131/13708745)):
1. Go to Pycharm > Preferences
2. Project: [insert project name here] > Python Interpreter
3. Next to your Python interpreter, click the down facing arrow and select "Show all"
4. In that Menu, highlight your interpreter and then in the right menu, select the button "Show paths for the selected interpreter" (this is the last button)
5. Click the plus symbol to add your path