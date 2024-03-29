# ReNom

Documents are available on the ReNom.jp web site.

- http://renom.jp/index.html

## ReNom version 2.7

- http://renom.jp/packages/renomdl/index.html

#### Changes from 2.6

Please refer to `changes` at renom.jp.

- http://renom.jp/packages/renomdl/rsts/change_history/main.html


## Requirements

- python 2.7, 3.4, 3.5, 3.6
- cuda-toolkit 8.0, 9.0, 9.1, 9.2, 10.0
- cudnn 7.0 ~ 7.4

For required python modules please refer to the requirements.txt.

## Installation

First clone the ReNom repository.

	git clone https://github.com/ReNom-dev-team/ReNom.git

Then move to the ReNom folder, install the module using pip.

	cd ReNom
  pip install -r requirements.txt
	pip install -e .

To activate CUDA, you have to build cuda modules before `pip install -e .` 
using following command.

    python setup.py build_ext -if

Please be sure that the environment variable CUDA_HOME is set correctly.

Example:

	$ echo $CUDA_HOME
	/usr/local/cuda-9.1
	
#### Wheels for linux environments

You can install ReNom using following wheel packages.  
Please download wheel for your environment and install it using pip command.

|   OS      |Python   |Cuda    |CuDnn   | Wheel |
|-----------|---------|--------|--------|-------|
|Ubuntu16.04|Python3.5|cuda9.2 |cudnn7.6|[v2.7.1](https://renom.jp/docs/downloads/wheels/renom_dl/ubuntu1604/python3.5/cuda9.2/cudnn7.6.2.24/ReNom2.7.1/renom-2.7.1-cp35-cp35m-linux_x86_64.whl)|
|           |         |        |        |[v2.7.3](https://renom.jp/docs/downloads/wheels/renom_dl/ubuntu1604/python3.5/cuda9.2/cudnn7.6.2.24/ReNom2.7.3/renom-2.7.3-cp35-cp35m-linux_x86_64.whl)|
|           |         |        |        |[v2.7.4](http://renom.jp/docs/downloads/wheels/renom_dl/ubuntu1604/python3.5/cuda9.2/cudnn7.6.2.24/ReNom2.7.4/renom-2.7.4_cuda9.2-cp35-cp35m-linux_x86_64.whl)|
|           |         |        |        |[v2.7.5](http://renom.jp/docs/downloads/wheels/renom_dl/ubuntu1604/python3.5/cuda9.2/cudnn7.6.2.24/ReNom2.7.5/renom-2.7.5_cuda9.2-cp35-cp35m-linux_x86_64.whl)|
|           |         |cuda10.0|cudnn7.6|[v2.7.1](https://renom.jp/docs/downloads/wheels/renom_dl/ubuntu1604/python3.5/cuda10.0/cudnn7.6.2.24/ReNom2.7.1/renom-2.7.1-cp35-cp35m-linux_x86_64.whl)|
|           |         |        |        |[v2.7.3](https://renom.jp/docs/downloads/wheels/renom_dl/ubuntu1604/python3.5/cuda10.0/cudnn7.6.2.24/ReNom2.7.3/renom-2.7.3-cp35-cp35m-linux_x86_64.whl)|
|           |         |        |        |[v2.7.4](http://renom.jp/docs/downloads/wheels/renom_dl/ubuntu1604/python3.5/cuda10.0/cudnn7.6.2.24/ReNom2.7.4/renom-2.7.4_cuda10.0-cp35-cp35m-linux_x86_64.whl)|
|           |         |        |        |[v2.7.5](http://renom.jp/docs/downloads/wheels/renom_dl/ubuntu1604/python3.5/cuda10.0/cudnn7.6.2.24/ReNom2.7.5/renom-2.7.5_cuda10.0-cp35-cp35m-linux_x86_64.whl)|
|           |Python3.6|cuda9.2 |cudnn7.6|[v2.7.1](https://renom.jp/docs/downloads/wheels/renom_dl/ubuntu1604/python3.6/cuda9.2/cudnn7.6.2.24/ReNom2.7.1/renom-2.7.1-cp36-cp36m-linux_x86_64.whl)|
|           |         |        |        |[v2.7.3](https://renom.jp/docs/downloads/wheels/renom_dl/ubuntu1604/python3.6/cuda9.2/cudnn7.6.2.24/ReNom2.7.3/renom-2.7.3-cp36-cp36m-linux_x86_64.whl)|
|           |         |        |        |[v2.7.4](http://renom.jp/docs/downloads/wheels/renom_dl/ubuntu1604/python3.6/cuda9.2/cudnn7.6.2.24/ReNom2.7.4/renom-2.7.4_cuda9.2-cp36-cp36m-linux_x86_64.whl)|
|           |         |        |        |[v2.7.5](http://renom.jp/docs/downloads/wheels/renom_dl/ubuntu1604/python3.6/cuda9.2/cudnn7.6.2.24/ReNom2.7.5/renom-2.7.5_cuda9.2-cp36-cp36m-linux_x86_64.whl)|
|           |         |cuda10.0|cudnn7.6|[v2.7.1](https://renom.jp/docs/downloads/wheels/renom_dl/ubuntu1604/python3.6/cuda10.0/cudnn7.6.2.24/ReNom2.7.1/renom-2.7.1-cp36-cp36m-linux_x86_64.whl)|
|           |         |        |        |[v2.7.3](https://renom.jp/docs/downloads/wheels/renom_dl/ubuntu1604/python3.6/cuda10.0/cudnn7.6.2.24/ReNom2.7.3/renom-2.7.3-cp36-cp36m-linux_x86_64.whl)|
|           |         |        |        |[v2.7.4](http://renom.jp/docs/downloads/wheels/renom_dl/ubuntu1604/python3.6/cuda10.0/cudnn7.6.2.24/ReNom2.7.4/renom-2.7.4_cuda10.0-cp36-cp36m-linux_x86_64.whl)|
|           |         |        |        |[v2.7.5](http://renom.jp/docs/downloads/wheels/renom_dl/ubuntu1604/python3.6/cuda10.0/cudnn7.6.2.24/ReNom2.7.5/renom-2.7.5_cuda10.0-cp36-cp36m-linux_x86_64.whl)|
|Ubuntu18.04|Python3.5|cuda9.2 |cudnn7.6|[v2.7.1](https://renom.jp/docs/downloads/wheels/renom_dl/ubuntu1804/python3.5/cuda9.2/cudnn7.6.2.24/ReNom2.7.1/renom-2.7.1-cp35-cp35m-linux_x86_64.whl)|
|           |         |        |        |[v2.7.3](https://renom.jp/docs/downloads/wheels/renom_dl/ubuntu1804/python3.5/cuda9.2/cudnn7.6.2.24/ReNom2.7.3/renom-2.7.3-cp35-cp35m-linux_x86_64.whl)|
|           |         |        |        |[v2.7.4](http://renom.jp/docs/downloads/wheels/renom_dl/ubuntu1804/python3.5/cuda9.2/cudnn7.6.2.24/ReNom2.7.4/renom-2.7.4_cuda9.2-cp35-cp35m-linux_x86_64.whl)|
|           |         |        |        |[v2.7.5](http://renom.jp/docs/downloads/wheels/renom_dl/ubuntu1804/python3.5/cuda9.2/cudnn7.6.2.24/ReNom2.7.5/renom-2.7.5_cuda9.2-cp35-cp35m-linux_x86_64.whl)|
|           |         |cuda10.0|cudnn7.6|[v2.7.1](https://renom.jp/docs/downloads/wheels/renom_dl/ubuntu1804/python3.5/cuda10.0/cudnn7.6.2.24/ReNom2.7.1/renom-2.7.1-cp35-cp35m-linux_x86_64.whl)|
|           |         |        |        |[v2.7.3](https://renom.jp/docs/downloads/wheels/renom_dl/ubuntu1804/python3.5/cuda10.0/cudnn7.6.2.24/ReNom2.7.3/renom-2.7.3-cp35-cp35m-linux_x86_64.whl)|
|           |         |        |        |[v2.7.4](http://renom.jp/docs/downloads/wheels/renom_dl/ubuntu1804/python3.5/cuda10.0/cudnn7.6.2.24/ReNom2.7.4/renom-2.7.4_cuda10.0-cp35-cp35m-linux_x86_64.whl)|
|           |         |        |        |[v2.7.5](http://renom.jp/docs/downloads/wheels/renom_dl/ubuntu1804/python3.5/cuda10.0/cudnn7.6.2.24/ReNom2.7.5/renom-2.7.5_cuda10.0-cp35-cp35m-linux_x86_64.whl)|
|           |Python3.6|cuda9.2 |cudnn7.6|[v2.7.1](https://renom.jp/docs/downloads/wheels/renom_dl/ubuntu1804/python3.6/cuda9.2/cudnn7.6.2.24/ReNom2.7.1/renom-2.7.1-cp36-cp36m-linux_x86_64.whl)|
|           |         |        |        |[v2.7.3](https://renom.jp/docs/downloads/wheels/renom_dl/ubuntu1804/python3.6/cuda9.2/cudnn7.6.2.24/ReNom2.7.3/renom-2.7.3-cp36-cp36m-linux_x86_64.whl)|
|           |         |        |        |[v2.7.4](http://renom.jp/docs/downloads/wheels/renom_dl/ubuntu1804/python3.6/cuda9.2/cudnn7.6.2.24/ReNom2.7.4/renom-2.7.4_cuda9.2-cp36-cp36m-linux_x86_64.whl)|
|           |         |        |        |[v2.7.5](http://renom.jp/docs/downloads/wheels/renom_dl/ubuntu1804/python3.6/cuda9.2/cudnn7.6.2.24/ReNom2.7.5/renom-2.7.5_cuda9.2-cp36-cp36m-linux_x86_64.whl)|
|           |         |cuda10.0|cudnn7.6|[v2.7.1](https://renom.jp/docs/downloads/wheels/renom_dl/ubuntu1804/python3.6/cuda10.0/cudnn7.6.2.24/ReNom2.7.1/renom-2.7.1-cp36-cp36m-linux_x86_64.whl)|
|           |         |        |        |[v2.7.3](https://renom.jp/docs/downloads/wheels/renom_dl/ubuntu1804/python3.6/cuda10.0/cudnn7.6.2.24/ReNom2.7.3/renom-2.7.3-cp36-cp36m-linux_x86_64.whl)|
|           |         |        |        |[v2.7.4](http://renom.jp/docs/downloads/wheels/renom_dl/ubuntu1804/python3.6/cuda10.0/cudnn7.6.2.24/ReNom2.7.4/renom-2.7.4_cuda10.0-cp36-cp36m-linux_x86_64.whl)|
|           |         |        |        |[v2.7.5](http://renom.jp/docs/downloads/wheels/renom_dl/ubuntu1804/python3.6/cuda10.0/cudnn7.6.2.24/ReNom2.7.5/renom-2.7.5_cuda10.0-cp36-cp36m-linux_x86_64.whl)|


Example: 

    # Download a wheel then run following command.
    pip install renom-2.7.3-cp35-cp35m-linux_x86_64.whl


## Precision

If you set an environment variable RENOM_PRECISION=64, 
calculations are performed with float64.

Default case, the precision is float32.

## Limit of tensor dimension size.
In ReNom version >= 2.4, only tensors that have less than 6 dimension size can be operated.


## License

“ReNom” is provided by GRID inc., as subscribed software.  By downloading ReNom, you are agreeing to be bound by our ReNom Subscription agreement between you and GRID inc.
To use ReNom for commercial purposes, you must first obtain a paid license. Please contact us or one of our resellers.  If you are an individual wishing to use ReNom for academic, educational and/or product evaluation purposes, you may use ReNom royalty-free.
The ReNom Subscription agreements are subject to change without notice. You agree to be bound by any such revisions. You are responsible for visiting www.renom.jp to determine the latest terms to which you are bound.
