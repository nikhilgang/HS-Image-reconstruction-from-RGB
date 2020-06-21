NV_GPU=GPU-8bfe8998-2454-ea30-f9cd-0d6b1c8abf08 nvidia-docker run -it -v /home/dgxuser107/:/home/dgxuser107/data nvcr.io/nvidia/tensorflow:17.11
cd ..
cd home
cd dgxuser107/
cd data/ 
pip install matplotlib
pip install pandas
pip install keras
pip install --upgrade keras==2.1.3
pip install pillow
