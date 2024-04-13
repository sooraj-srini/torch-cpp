sudo apt install nvtop unzip htop
if [ ! -d "libtorch" ]; then
wget https://download.pytorch.org/libtorch/nightly/cu121/libtorch-cxx11-abi-shared-with-deps-latest.zip
unzip libtorch-cxx11-abi-shared-with-deps-latest.zip
fi
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=/root/sem/torch-cpp/libtorch/ ..