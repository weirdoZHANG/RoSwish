# RoSwish: A novel Rotating Swish activation function with adaptive rotation around zero (一种新颖的Rotating Swish激活函数，具有围绕零点的自适应旋转特性)
This software allows users to reproduce the results in RoSwish, Shengbo Zhang and Guoqiang Ren '2025'.

# RoSwish Mathematical Definitions
The mathematical definition adopted in this paper is `(x + α) * sigmoid(β * x) - 0.5 * α`, where `α` and `β` represent the trainable parameters in the hidden layer dimension, which together regulate the positive slope, negative saturation, and negative concavity. And `α` and `β` start learning from `0.817` and `3`, respectively. `x * sigmoid(β * x) + 0.5 * α * tanh(0.5 * β * x)` is the equivalent transformation of the former, but it has high computational complexity.

# NsSinAFs Mathematical Definitions
The general mathematical definition is `NsSinAFs(f)`, where `f = a * x + b * sin(x)`, where `a` and `b` represent the trainable parameters in the hidden layer dimension, and `a` and `b` start learning from `1` and `0`, respectively. In layman's terms, `NsSinAFs(f)` replaces the input `x` of the original `AFs(x)` with `f`. `f` is equivalent to adding the `sin` function to the input `x` of the original `AFs(x)` and multiplying the trainable parameters `a` and `b`, respectively.

# Tip
Based on the simple tasks and models considered in this study, we temporarily set the initial values of parameters `α`, `β`, `a`, and `b` to `0.817`, `3`, `1`, and `0`, respectively. Suppose researchers find better initial values of the parameters and can freely adjust them.

# Execution
Please install torch, pandas, sklearn and Python 3+.

# Citation
If you find this useful in your research, please consider citing:

    @article{ZHANG2025107892,
      title = {RoSwish: A novel Rotating Swish activation function with adaptive rotation around zero},
      author = {Shengbo Zhang and Guoqiang Ren},
      journal = {Neural Networks},
      year = {2025},
      pages = {107892},
      issn = {0893-6080},
      doi = {https://doi.org/10.1016/j.neunet.2025.107892},
      url = {https://www.sciencedirect.com/science/article/pii/S0893608025007737},
    }

# Contact
If you have any questions or want to use the code, please contact zhangshengbo2049@outlook.com.
