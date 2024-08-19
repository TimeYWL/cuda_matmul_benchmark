// An example

#include <iostream>
#include <vector>
#include "simulator.h"
#include "operator.h"
#include "utils.h"

int main(int argc, char* argv[]) {
	std::vector<OpConfig> configVector;
	
	// OpConfig conv2dConfig;
	// conv2dConfig.opType = OpType::CONV2D;
	// conv2dConfig.args   = {
	// 	/* Group_count = */ 1,
	// 	/* Batch_size  = */ 64,
	// 	/* In_channels_per_grp =  */ 64,
	// 	/* In_h        = */ 28,
	// 	/* In_w        = */ 28,
	// 	/* Out_channels= */ 128,
	// 	/* Kn_h        = */ 3,
	// 	/* Kn_w        = */ 3,
	// 	/* Pad_h       = */ 1,
	// 	/* Pad_w       = */ 1,
	// 	/* Stride_h    = */ 1,
	// 	/* Stride_w    = */ 1,
	// 	/* Dila_h      = */ 1,
	// 	/* Dila_w      = */ 1
	// };

	OpConfig matMulConfig;
	matMulConfig.opType = OpType::MATMUL;
	int M = 512;
	int N = 4096;
	int K = 4096;
	matMulConfig.args   = {
		/* M = */ M,
		/* N = */ N,
		/* K = */ K,
		/* transa = */ 0,
		/* transb = */ 0,
		/* tenser_op = */ 0,
		/* algo   = */ -1,
		/* dtype*/ DTYPE_HALF
	};

	//configVector.push_back(conv2dConfig);
	configVector.push_back(matMulConfig);

	std::cout << "Test performence of Gemm\n";
	std::cout << "M=" << M << " N=" << N << " K=" << K << std::endl;
	Simulator* simu = new Simulator;
	simu->initOp(configVector);
	PfMap pfMap = simu->measureAllOp(1000);
	printf("Avg time: %F ms  TFLOPS: %F\n", pfMap[matMulConfig].getDurtime(), pfMap[matMulConfig].getTflops());
	simu->freeOp(); 
	delete simu;

	std::cout << "End of test\n";
	return 0;
}
