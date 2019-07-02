# Reproduction Code fCMA-ES
This repository contains code to reproduce the main results of "Large-Scale Noise-.Resilient Evolution Strategies". 

It requires the Shark 4.1 development branch: https://github.com/Shark-ML/Shark/tree/4.1
If you installed Shark in /home/user/Shark/build, clone this repo and run

```
mkdir build
cd build
cmake -DShark_DIR=/home/user/Shark/build ../
make -j2
./ExampleRuns
./Runtime
```

the first program will generate the single run experiments, the second the noise-free experiments with varying dimensions. Results will be stored as CSV files in `build/results`.
The experiments will take some time, but programs will run multi-threaded by default.