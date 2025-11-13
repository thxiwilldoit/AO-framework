# AO-framework

This is the code for our paper titled: 'Visual-Enhanced Multimodal Framework for Flexible Job Shop Scheduling Problem', which has been accepted by ACM MM 2025 (2025 ACM Multimedia). We encourage everyone to use this code and cite our paper:

```
@inproceedings{zhao2025visual,
  title={Visual-Enhanced Multimodal Framework for Flexible Job Shop Scheduling Problem},
  author={Zhao, Peng and Cao, Zhiguang and Wang, Di and Song, Wen and Pang, Wei and Zhou, You and Jiang, Yuan},
  booktitle={Proceedings of the 33rd ACM International Conference on Multimedia},
  pages={2496--2505},
  year={2025}
}
```

### Introduction

`ini_graph.py` is used to generate visual information (images) for scheduling examples. The tensor representing scheduling instances needs to include data with the following dimensions:

	*num_job*: number of jobs
    *num_mas*: number of machines
    *num_eve_ope*: the tensor that contains the number of operations per jobs
    *ope_ma_adj_batch*: the tensor that can process 0/1 machines per operation
    *sequence_batch* is all -1 by default, and the length is the number of operations

We have improved the code's execution efficiency by generating this in memory. Additionally, this file provides a set of FJSP example data with a size of $3 \times 3$ for testing the code's performance.


`img_net.py` includes the functions and networks required by the AO framework. It's important to note that applying AO to different codes requires different network parameters (i.e., the input dimensions of each network), which need to be adjusted according to the dimensions defined in the current code. This is due to the varying feature dimensions defined by different FJSP solvers.

The *rotate_vectors* function is not utilized in the actual code; it originated from a previous idea of generating specific angles to control the rotation of commonness through network processing of features. However, it has been found to be inefficient and ineffective. We have retained it for cross-validation of the theory in Appendix B and for testing purposes for those who may need it.


`process.py` contains the computational process of the AO framework. In practical applications, it should be integrated into the feature processing section of the target FJSP solver.

