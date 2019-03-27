Introduction
=============

ReNomDL is a Deep Learning frame work.
ReNom DL version 3 is available.

Concept of ReNom version 3
---------------------------

The goal of ReNomDL version 3.0 is to allow users implement high performance neural networks with 
minimal programming skills and device-related issues. Many deep learning frame works are provided from 
many vendors, however most of these frameworks depends a lot on the programming skills of the user, 
either disallowing beginners entry to the more optimized libraries or providing little flexibility in 
the easier frameworks. This impacts the training and inference times of your networks significantly and 
users desiring more performance and flexibility from their frameworks must eventually deal with 
learning cuda-device programming, parallel computing, etc. ReNomDL serves as a mediator between user-defined 
neural networks  and their devices, allowing for high speed and good scalability in their own machines.



Features
---------

ReNomDL allows you to implement neural networks that are high performance at training speed and less gpu mempry.


Eager mode and Executor mode
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One of the features of ReNomDL is that it provides 2 modes for running neural networks. The first one, 
Eager mode, is the same as running your models in ReNomDL version 2. In Eager mode, the input data will 
flow through the layers and the output of each layer can be observed evaluated immediately.
The second one is Executor mode which is a new feature of ReNomDL version 3. 
You can extract a computational graph as an Executor object from the output of a neural network. 
This allows the Executor object to assume that the user is finished defining the graph and optimizes 
it according to the type of graph that was created.

Eager mode stores the user defined graph in information that can later be used in the executor mode to 
ensure that, while the user checks that the graph produces the correct results. Later on when finished 
building the graph, ReNomDL then has the option to execute the graph quickly using the executor mode.


.. code-block:: python

    import time
    import numpy as np
    import renom.graph as rmg
    from renom.cuda import set_cuda_active
    set_cuda_active(True)
    
    epoch = 5
    batch_size = 256
    opt = rmg.Sgd(0.001)
    x = np.random.randn(16384, 100)
    y = np.random.randn(16384, 10)
    
    dense1 = rmg.Dense(1000)
    dense2 = rmg.Dense(10)
    loss_function = rmg.MeanSquared()
    
    def NN(x, y):
        h = rmg.relu((dense1(x)))
        h = dense2(h)
        l = loss_function(h, y)
        return l, h
    
    ### Eager Mode
    start_t = time.time()
    for e in range(epoch):
        batch_loop = len(x)//batch_size
        perm = np.random.permutation(len(x))
        for b in range(batch_loop):
            batch_x = x[perm[b*batch_size:(b+1)*batch_size]]
            batch_y = y[perm[b*batch_size:(b+1)*batch_size]]
            loss, _ = NN(batch_x, batch_y)
            loss.backward()
            loss.update(opt)
    print("%3.2f [sec]"%(time.time() - start_t))
    >>> 3.93 [sec]
    
    ## Executor Mode
    dispatcher = rmg.Distro(x, y, batch_size=batch_size, num_gpus=1, shuffle=True, drop_remainder=True)
    loss, p = NN(*dispatcher.get_output_graphs())
    exc = loss.get_executor(mode='training', optimizer=opt, with_validation=False)
    
    start_t = time.time()
    exc.execute(epochs=epoch, progress=False)
    print("%3.2f [sec]"%(time.time() - start_t))
    >>> 1.45 [sec]


Computational Graph optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One of the key features of ReNomDL v3 is that it now produces a computational graph, whereas in v2, 
the graph existed as a result of the user connecting the different computational elements together. 
As of v3, this connection is now made explicit in the graph, giving it the opportunity to make predictions on 
the operations that are to take place during execution mode. A key advantage of this is that it allows 
the graph to reduce the required space to a minimum, allows us to remove unnecessary operations or simply 
use better algorithms to perform the same computations as before the optimization.

All of this happens in a much more light-weight form as part of the optimization procedure reduces 
execution to the bare minimum of what is actually required to perform the graph, 
which becomes significantly more important as multi-device machines are becoming more and more important


