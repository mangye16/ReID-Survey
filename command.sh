conda create -n reid python=3.6
conda activate reid
pip install yacs
pip install -U scipy==1.2.0
conda install -c pytorch pytorch torchvision ignite==0.1.2 

conda install ipykernel


# source activate reid
# python -m ipykernel install --user --name REID --display-name "Python (REID)"