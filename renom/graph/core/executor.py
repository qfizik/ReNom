import numpy as np
from tqdm import tqdm
import renom as rm


def _norm_init(info):
    info['nth_epoch'] = 0
    info['all_losses'] = []


def _norm_epoch_start(info):
    info['loss'] = 0
    info['bar'] = tqdm(total=len(info['inputs'][0]))
    info['epoch_loss_list'] = []
    for disp in info['inputs']:
        disp.reset()


def _norm_step_finish(info):
    loss = info['losses']
    epoch_loss_list = info['epoch_loss_list']
    bar = info['bar']
    nth_epoch = info['nth_epoch']

    loss = float(loss[0].as_ndarray())
    epoch_loss_list.append(loss)
    bar.set_description("epoch={:03d} cur-loss={:5.3f}".format(nth_epoch, loss))
    bar.update(1)


def _norm_epoch_finish(info):
    epoch_loss_list = info['epoch_loss_list']
    bar = info['bar']
    all_losses = info['all_losses']
    nth_epoch = info['nth_epoch']

    epoch_loss_list.pop(-1)
    all_losses.append(np.sum(epoch_loss_list))
    bar.n = bar.n - 1
    bar.set_description(
        "epoch={:03d} avg-loss={:5.3f}".format(nth_epoch, np.mean(epoch_loss_list)))
    bar.close()
    info['nth_epoch'] += 1


def _validation_func(data, target, val_dict):
    def _perform_validation(info):
        ins = info['inputs']
        losses = info['losses']

        norm_d, norm_t = ins[0].value, ins[1].value
        ins[0].value, ins[1].value = data, target
        ins[1]._perm = ins[0]._perm
        bar = tqdm(total=len(ins[0]))
        for i in ins:
            i.reset()
        lst = []
        try:
            while(True):
                for depth in val_dict.keys():
                    for call in val_dict[depth]:
                        call.perform()
                loss = float(losses[0].as_ndarray())
                bar.set_description('validation cur-loss={:5.3f}'.format(loss))
                bar.update(1)
                lst.append(loss)
        except StopIteration:
            lst.pop(-1)
            bar.set_description('validation avg-loss={:5.3f}'.format(np.mean(lst)))
            bar.close()
            ins[0].value, ins[1].value = norm_d, norm_t
            ins[1]._perm = ins[0]._perm
            info['validation_loss'] = np.sum(lst)
        # _perform_validation END
    return _perform_validation


class Executor:
    '''
      The Executor class is ...

      Args:
          call_list (list):
          graph_element (GraphElement):
          losses (GraphElement):
    '''

    def __init__(self, call_list, special_ops):
        self.call_list = call_list
        self.dispatchers = special_ops['graph_inputs']
        self.loss = special_ops['losses']

        self._events = {'Initialize': [],
                        'Epoch-Start': [],
                        'Step-Start': [],
                        'Step-Finish': [],
                        'Epoch-Finish': [],
                        'Teardown': [],
                        }

        self.register_event('Initialize', _norm_init)
        self.register_event('Epoch-Start', _norm_epoch_start)
        self.register_event('Step-Finish', _norm_step_finish)
        self.register_event('Epoch-Finish', _norm_epoch_finish)

    def execute(self, epochs, progress=True):
        '''
          This function executes computational graph.

          Args:
              epochs (int): Number of epochs.
              progress (bool): If True is given, the progress will be shown.
        '''
        exe_info = {'inputs': self.dispatchers,
                    'losses': self.loss,
                    'progress': progress
                    }
        for ev in self._events['Initialize']:
            ev(exe_info)

        for e in range(epochs):
            self.perform_event_epoch(exe_info)

        for ev in self._events['Teardown']:
            ev(exe_info)

        return exe_info['all_losses']

    def __del__(self):
        for i in range(len(self.dispatchers)):
            self.dispatchers[i] = None
        for i in range(len(self.loss)):
            self.loss[i] = None

    def perform_event_epoch(self, exe_info):
        for ev in self._events['Epoch-Start']:
            ev(exe_info)
        try:
            while(True):
                self.perform_event_step(exe_info)
        except StopIteration:
            pass
        for ev in self._events['Epoch-Finish']:
            ev(exe_info)
        return

    def perform_epoch(self):
        try:
            while(True):
                self.perform_step()
        except StopIteration:
            pass

    def perform_event_step(self, exe_info):
        for ev in self._events['Step-Start']:
            ev(exe_info)
        for depth in self.call_list.keys():
            for call in self.call_list[depth]:
                if rm.logging_level >= 10:
                    call.logged_perform()
                else:
                    call.perform()
        for ev in self._events['Step-Finish']:
            ev(exe_info)

    def perform_step(self):
        for depth in self.call_list.keys():
            for call in self.call_list[depth]:
                if rm.logging_level >= 10:
                    call.logged_perform()
                else:
                    call.perform()

    def register_event(self, event_name, event_function):
        assert isinstance(event_name, str) and event_name in self._events
        assert callable(event_function)
        self._events[event_name].append(event_function)

    def _set_validation(self, val_data, val_target, val_dct):
        self.register_event('Epoch-Finish', _validation_func(val_data, val_target, val_dct))

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
        self.dispatchers[1]._perm = self.dispatchers[0]._perm
