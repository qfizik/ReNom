import numpy as np
from tqdm import tqdm
import renom as rm


def _norm_init(info):
    info['nth_epoch'] = 0
    info['all_losses'] = []


def _norm_epoch_start(info):
    info['bar'] = tqdm(total=len(info['inputs'][0]))
    if info['mode'] == 'training':
        info['epoch_name'] = 'Epoch={:03d}'.format(info['nth_epoch'])
    else:
        info['epoch_name'] = 'Validation'
    info['epoch_loss_list'] = []
    for disp in info['inputs']:
        disp.reset()


def _norm_step_finish(info):
    loss = info['losses']
    if info['mode'] == 'step':
        info['step_loss'] = float(loss[0].as_ndarray())
        return
    epoch_loss_list = info['epoch_loss_list']
    bar = info['bar']
    epoch_name = info['epoch_name']

    loss = float(loss[0].as_ndarray())
    epoch_loss_list.append(loss)
    bar.set_description("{0!s: >10} cur-loss={1:5.3f}".format(epoch_name, loss))
    bar.update(1)


def _norm_epoch_finish(info):
    epoch_loss_list = info['epoch_loss_list']
    bar = info['bar']
    all_losses = info['all_losses']
    epoch_name = info['epoch_name']

    epoch_loss_list.pop(-1)
    all_losses.append(np.sum(epoch_loss_list))
    bar.set_description(
        "{0!s: >10} avg-loss={1:5.3f}".format(epoch_name, np.mean(epoch_loss_list)))
    bar.close()
    info['nth_epoch'] += 1


def _validation_func(data, target):
    def _perform_validation(info):
        #TODO: Move this to event
        ins = info['inputs']
        if info['mode'] == 'training':
            losses = info['losses']
            norm_d, norm_t = ins[0].value, ins[1].value
            ins[0].value, ins[1].value = data, target
            ins[1]._perm = ins[0]._perm
            info['epoch_loss_list'] = []
            info['mode'] = 'inference'
            info['norm_data'] = (norm_d, norm_t)
            info['nth_epoch'] -= 1 # Redo the epoch as validation
        else:
            norms = info['norm_data']
            info['mode'] = 'training'
            ins[0].value, ins[1].value = norms[0], norms[1]
            ins[1]._perm = ins[0]._perm
            info['validation_loss'] = np.sum(info['epoch_loss_list'])
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

    def __init__(self, call_list, special_ops, mode='inference'):
        self.call_list = call_list
        self.dispatchers = special_ops['graph_inputs']
        self.loss = special_ops['losses']
        self.mode = mode

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
                    'progress': progress,
                    'mode': self.mode,
                    }
        for ev in self._events['Initialize']:
            ev(exe_info)

        while exe_info['nth_epoch'] < epochs:
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

    def perform_event_step(self, exe_info):
        for ev in self._events['Step-Start']:
            ev(exe_info)

        mode = exe_info['mode']
        assert isinstance(self.call_list, dict)
        if mode == 'inference' or mode == 'step':
            parts = ['Forward']
        elif mode == 'training':
            parts = ['Forward', 'Backward', 'Gradient']
        else:
            raise NotImplementedError()
        for part in parts:
            for depth in sorted(self.call_list[part].keys()):
                for call in self.call_list[part][depth]:
                    if rm.logging_level >= 10:
                        call.logged_perform()
                    else:
                        call.perform()

        for ev in self._events['Step-Finish']:
            ev(exe_info)

    def register_event(self, event_name, event_function):
        assert isinstance(event_name, str) and event_name in self._events
        assert callable(event_function)
        self._events[event_name].append(event_function)

    def unregister_events(self, event_name):
        assert isinstance(event_name, str) and event_name in self._events
        self._events[event_name] = []

    def _set_validation(self, val_data, val_target):
        self.register_event('Epoch-Finish', _validation_func(val_data, val_target))

    def step(self, d, t):
        #TODO: Clean up this mess boy.
        exe_info = {
            'mode': 'step',
            'losses': self.loss,
        }
        ina = self.dispatchers[0]
        inb = self.dispatchers[1]
        p_d, p_t = ina.value, inb.value
        ina.value, inb.value = d, t
        inb._perm = ina._perm
        self.perform_event_step(exe_info)
        loss = self.loss[0].as_ndarray()
        ina.value, inb.value = p_d, p_t
        inb._perm = ina._perm
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
