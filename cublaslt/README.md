运行方式：

```bash
# 编译
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j

# 运行 (会打印所有algo的结果并按性能排序，第一个为最好结果)
./sample_cublasLt_LtSgemmCustomFind 
```

可在 `main.cpp` 中更改矩阵乘的形状（M, N, K），以及数据类型（DTYPE_FLOAT 或 DTYPE_HALF），更改后需重新编译。
