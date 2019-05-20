Assessing the distribution of recombination proteins during C. *elegans* meiotic progression - Chloe Girard from Anne Villeneuve lab
==================================

## Setup
1 - Download [Anaconda](https://www.anaconda.com/download/), a free installer that includes Python and all the common scientific packages.

2- Clone (or download) the repository

(On Window, you may need to install [Git](https://hackernoon.com/install-git-on-windows-9acf2a1944f0))

```
git clone https://github.com/bioimage-analysis/Chloe_Chromosome_detection
```

3- Go into the directory

```
cd Chloe_Chromosome_detection/
```

4- Create a conda environment with an Ipython kernel:

(To install python / conda on a Window environment you can follow this [LINK](https://medium.com/@GalarnykMichael/install-python-on-windows-anaconda-c63c7c3d1444))

```
 conda create --name name_env python=3 ipykernel
```

5- Activate your conda environment:

```
source activate name_env
```

6- Install dependency from the requirements.txt :

```
pip install -r requirements.txt
```
(The installation of python-bioformats / javabridge might fail, if so, download Java Dev. kit, [here](https://www.oracle.com/technetwork/java/javase/downloads/jdk8-downloads-2133151.html))

## Usage

cd to the notebook directory and lunch jupyter notebook:

```
jupyter notebook
```
You will find a basic introduction on how to use the Jupyter Notebook [HERE](https://jupyter-notebook.readthedocs.io/en/stable/examples/Notebook/Notebook%20Basics.html)

## Note about Notebook in github

If it doesn't load try using this [link](https://nbviewer.jupyter.org/).

## Contact
Cedric Espenel  
E-mail: espenel@stanford.edu
