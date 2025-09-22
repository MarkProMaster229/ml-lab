# ml-lab
**https://github.com/MarkProMaster229/ml-lab/wiki**

cd /mnt/storage/product/ml-lab/baseModel
g++ -std=c++17 -g \
-IGeneration -IModel -ICore \
Core/main.cpp Core/Runner.cpp Core/BatchGenerator.cpp Model/Transformer.cpp \
-o Core/main.out
