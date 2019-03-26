Introduction
=============

ReNomDL is a Deep Learning frame work.
ReNom DL version 3 is available.

Concept of ReNom version 3
---------------------------

The goal of ReNomDL version 3.0 is to allow users implement a high performance neural network.
So far, many deep learning frame works are provided from many vendors. However many of frame works 
requires users programming skills. The training time and inference time is depends on users programming skill
that are including cuda programming, parallel computing and so on.

ReNomDL has functions to optimize user defined neural networks. This allows users high speed and scalable training.


High Performance
~~~~~~~~~~~~~~~~~

ReNomDL allows you to implement neural networks that are high performance at training speed and less gpu mempry.


Eager mode and Executor mode
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ReNomDL provides 2 modes to run neural network. The first one, **Eager mode**, is same running mode as 
ReNomDL version 2. In **Eager mode**, the input data will flow layers and the output of each layers are
evaluated immediately.

The second one is **Executor mode** which is new feature of ReNomDL version 3. You can extract computational
graph as an **Executor object** from outputs of neural network. The **Executor object** will be optimized according to
the information that the graph nodes has. This 


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

As mentioned above, `Executor mode` will optimize the extracted computational graph.



Multi GPU Scalability
~~~~~~~~~~~~~~~~~~~~~

Graph will be here.


