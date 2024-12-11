# MiniTorch Module 4

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module4.html

This module requires `fast_ops.py`, `cuda_ops.py`, `scalar.py`, `tensor_functions.py`, `tensor_data.py`, `tensor_ops.py`, `operators.py`, `module.py`, and `autodiff.py` from Module 3.


Additionally you will need to install and download the MNist library.

(On Mac, this may require installing the `wget` command)

```
pip install python-mnist
mnist_get_data.sh
```


* Tests:

```
python run_tests.py
```

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/tensor.py minitorch/datasets.py minitorch/testing.py minitorch/optim.py minitorch/tensor_ops.py minitorch/fast_ops.py minitorch/cuda_ops.py project/parallel_check.py tests/test_tensor_general.py

## Task 4.5: Sentiment Analysis (SST2)
- **Training Set Size**: 450
- **Validation Set Size**: 100
- **Learning Rate**: 0.01
- **Maximum Epochs**: 250

### Training Log:
- Ran up to **epoch 55**:
  - **Loss**: 10.51
  - **Train Accuracy**: 87.78%
  - **Validation Accuracy**: 71.00%
  - **Best Validation Accuracy**: 75.00%

Training log: [Click to View Sentiment Log](./sentiment.txt)

## Task 4.5: Digit classification (MNIST)
- **Training Set Size**: 5000 
- **Validation Set Size**: 500
- **Learning Rate**: 0.01
- **Maximum Epochs**: 500

### Training Log:
- Ran up to **epoch 500**:
  - **Loss**: 
  - **Train Accuracy**: 
  - **Validation Accuracy**: 
  - **Best Validation Accuracy**: 
Training log: [Click to View MNIST Log](./mnist.txt)