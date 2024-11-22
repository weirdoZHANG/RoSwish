# RoSwish: A novel Rotating Swish activation function with adaptive rotation around zero point
This software allows users to reproduce the results in RoSwish, Shengbo Zhang and Guoqiang Ren 2025.

# RoSwish Mathematical Definitions
The mathematical definition adopted in this paper is `(x + α) * sigmoid(β * x) - 0.5 * α`. `x * sigmoid(β * x) + 0.5 * α * tanh(0.5 * β * x)` is the equivalent transformation of the former, but it has high computational complexity.

# Execution
Please install pandas, sklearn and Python 3+.

## Citation
If you find this useful in your research, please consider citing:

    @article{,
      title={},
      author={},
      journal={},
      year={}
    }
