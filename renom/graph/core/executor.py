import numpy as np
from tqdm import tqdm
import renom as rm

class Executor:
    '''
      The Executor class is ...

      Args:
          call_list (list):
          graph_element (GraphElement):
          losses (GraphElement):
    '''

    def __init__(self, call_list, graph_inputs, losses):
        self.call_list = call_list
        self.dispatchers = graph_inputs
        self.loss = losses
        self._events = {}

    def execute(self, epochs, progress=True):
        '''
          This function executes computational graph.

          Args:
              epochs (int): Number of epochs.
              progress (bool): If True is given, the progress will be shown.
        '''
        # Preprocessing Start
        if 'Initialize' in self._events:
            self._events['Initialize'](self)
        nth_epoch = 0
        all_losses = []
        for disp in self.dispatchers:
            disp.reset()
        # Preprocessing End
        # Loop Start
        while(nth_epoch < epochs):
            try:
                # Init Epoch Start
                if 'Epoch-Start' in self._events:
                    self._events['Epoch-Start'](self)
                loss = 0
                if progress:
                    bar = tqdm()
                epoch_loss_list = []
                # Init Epoch End
                while(True):
                    # Single Step Start
                    if 'Step-Start' in self._events:
                        self._events['Step-Start'](self)
                    self.perform_step()
                    # Single Step End
                    # Retrieve Loss Start
                    if 'Loss-Start' in self._events:
                        self._events['Loss-Start'](self)
                    loss = float(self.loss[0].as_ndarray())
                    epoch_loss_list.append(loss)
                    if progress:
                        bar.set_description("epoch:{:03d} loss:{:5.3f}".format(nth_epoch, loss))
                        bar.update(1)
                    # Retrieve Loss End
            except StopIteration:
                # Epoch Finish Start
                if 'Epoch-Finish' in self._events:
                    self._events['Epoch-Finish'](self)
                epoch_loss_list.pop(-1)
                all_losses.append(np.sum(epoch_loss_list))
                for disp in self.dispatchers:
                    disp.reset()
                if progress:
                    bar.n = bar.n - 1
                    bar.set_description(
                        "epoch:{:03d} avg-loss:{:5.3f}".format(nth_epoch, np.mean(epoch_loss_list)))
                    bar.close()
                nth_epoch += 1
                # Epoch Finish End
        # Loop End
        return all_losses

    def __del__(self):
        for i in range(len(self.dispatchers)):
            self.dispatchers[i] = None
        for i in range(len(self.loss)):
            self.loss[i] = None

    def perform_step(self):
        for depth in self.call_list.keys():
            for call in self.call_list[depth]:
                if rm.logging_level >= 10:
                    call.logged_perform()
                else:
                    call.perform()

    def register_event(self, event_name, event_function):
        assert isinstance(event_name, str)
        assert callable(event_function)
        self._events[event_name] = event_function

    def step(self, d, t):
        self.dispatchers[0].value = d
        self.dispatchers[1].value = t
        self.perform_step()
        loss = self.loss[0].as_ndarray()
        return loss

    def set_input_data(self, data, target):
        assert len(self.dispatchers) == 2, 'This method assumes standard input methods'
        assert isinstance(data, np.ndarray) and isinstance(
            target, np.ndarray), 'The data should be given as NumPy arrays.'
        assert len(data) == len(target), 'Data and Target should have the same number of points'
        # TODO: These are magic numbers. There should be a convention for which
        # is which instead!
        self.dispatchers[0].value = data
        self.dispatchers[1].value = target
