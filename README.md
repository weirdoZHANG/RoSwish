# RoSwish: A novel Rotating Swish activation function with adaptive rotation around zero point
This software allows users to reproduce the results in RoSwish, Shengbo Zhang and Guoqiang Ren 2025.

# RoSwish Mathematical Definitions
The mathematical definition adopted in this paper is `(x + α) * sigmoid(β * x) - 0.5 * α`, where α and β represent the trainable parameters in the hidden layer dimension, which together regulate the positive slope, negative saturation, and negative concavity. And α and β start learning from 0.817 and 3, respectively. `x * sigmoid(β * x) + 0.5 * α * tanh(0.5 * β * x)` is the equivalent transformation of the former, but it has high computational complexity.

# NsSinAFs Mathematical Definitions
The general mathematical definition is `NsSinAFs(f)`, where `f = a * x + b * sin(x)`, where a and b represent the trainable parameters in the hidden layer dimension, and a and b start learning from 1 and 0, respectively.

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
