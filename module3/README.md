# MiniTorch Module 3

<img src="https://minitorch.github.io/_images/match.png" width="100px">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module3.html

This module requires `scalar.py`, `tensor_functions.py`, `tensor_data.py`, `tensor_ops.py`, `operators.py`, `module.py`, and `autodiff.py` from Module 2.

You will need to modify `tensor_functions.py` slightly in this assignment.

* Tests:

```
python run_tests.py
```

* Note:

Several of the tests for this assignment will only run if you are on a GPU machine and will not
run on github's test infrastructure. Please follow the instructions to setup up a colab machine
to run these tests.
This assignment requires the following files from the previous assignments.

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/scalar.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py


The output from parallel_check.py can be viewed [here](https://github.com/Cornell-Tech-ML/mle-module-3-ethanglaser/blob/master/parallel_output.txt).

### simple
![simple gpu results with runtime per epoc](imgs/simple.PNG)
![simple cpu results with runtime per epoc](imgs/simple_cpu.PNG)

### split
![split gpu results with runtime per epoc](imgs/split.PNG)
![split cpu results with runtime per epoc](imgs/split_cpu.PNG)

### xor
![xor gpu results with runtime per epoc](imgs/xor.PNG)
![xor cpu results with runtime per epoc](imgs/xor_cpu.PNG)