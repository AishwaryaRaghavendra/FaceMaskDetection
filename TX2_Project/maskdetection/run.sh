# sudo ln -s /usr/local/cuda-10.0/lib64/libcudart.so.10.0 /usr/local/cuda-10.0/lib64/libcudart.so.10.2
# sudo ln -s /usr/local/cuda-10.0/lib64/libcublas.so.10 /usr/local/cuda-10.0/lib64/libcublas.so.10
# sudo ln -s /usr/local/cuda-10.0/lib64/libcublasLt.so.10 /usr/local/cuda-10.0/lib64/libcublasLt.so.10
# sudo ln -s /usr/local/cuda-10.0/lib64/libcufft.so.10. /usr/local/cuda-10.0/lib64/libcufft.so.10
# sudo ln -s /usr/local/cuda-10.0/lib64/libcurand.so.10.0 /usr/local/cuda-10.0/lib64/libcurand.so.10
# sudo ln -s /usr/local/cuda-10.0/lib64/libcusolver.so.10.0 /usr/local/cuda-10.0/lib64/libcusolver.so.10
# sudo ln -s /usr/local/cuda-10.0/lib64/libcusparse.so.10.0 /usr/local/cuda-10.0/lib64/libcusparse.so.10
# sudo ln -s /usr/local/cuda-10.0/lib64/libcudnn.so.10.0 /usr/local/cuda-10.0/lib64/libcudnn.so.10


cd /usr/bin
sudo ./jetson_clocks
# source /home/nvidia/p3env/bin/activate
cd /home/nvidia/Desktop/TX2_Project/Social_Distancing_with_AI
python3 Run.py


# sudo rm /usr/local/cuda-10.0/lib64/libcublas.so.10.2