import numpy as np
from tqdm import tqdm
import renom as rm
import warnings
from contextlib import contextmanager


def _norm_init(info):
    info['nth_epoch'] = 0
    info['all_losses'] = []


def _norm_finish(info):
    if 'bar' in info:
        info['bar'].close()


def _norm_epoch_start(info):
    if info['mode'] == 'inference' and 'training_loss' in info:
        bar = info['bar']
        bar.total += len(info['inputs'][0])
        info['epoch_name'] = 'Validating'
    else:
        if 'bar' in info:
            info['bar'].close()
        info['bar'] = tqdm(total=len(info['inputs'][0]))
        info['epoch_name'] = 'Training'
    info['epoch_loss_list'] = []
    for disp in info['inputs']:
        disp.reset()


def _norm_step_finish(info):
    loss = info['losses']
    if len(loss) == 0:
        return
    if info['mode'] == 'step':
        if len(loss) > 0:
            info['step_loss'] = float(loss[-1].as_ndarray())
        return
    epoch_loss_list = info['epoch_loss_list']
    bar = info['bar']
    epoch_name = info['epoch_name']

    if len(loss) > 0:
        loss = float(loss[-1].as_ndarray())
        bar.set_description("{0!s: >10} cur-loss={1:5.3f}".format(epoch_name, loss))
        epoch_loss_list.append(loss)
    else:
        bar.set_description("{0!s: >10}".format(epoch_name))
    bar.update(1)


def _norm_epoch_finish(info):
    epoch_loss_list = info['epoch_loss_list']
    bar = info['bar']
    epoch_name = info['epoch_name']
    loss = info['losses']

    epoch_loss_list.pop(-1)
    # all_losses.append(np.sum(epoch_loss_list))
    cur_loss = np.mean(epoch_loss_list)
    if len(loss) == 0:
        bar.set_description("{0!s: >10}".format(epoch_name))
    elif info['mode'] == 'training':
        bar.set_description(
            "{0!s: >10} avg-loss={1:5.3f}".format(epoch_name, cur_loss))
        info['training_loss'] = cur_loss
    elif info['mode'] == 'inference' and 'training_loss' in info:
        epoch_name = 'Finished #{:03d}'.format(info['nth_epoch'])
        bar.set_description('{0!s: >10} [train={1:5.3f}, valid={2:5.3f}]'.format(
            epoch_name, info['training_loss'], cur_loss))
    # bar.close()
    info['nth_epoch'] += 1


def _validation_func():
    def _perform_validation(info):
        # TODO: Move this to event
        ins = info['inputs']
        if info['mode'] == 'training':
            # ins[0].switch_source(1)
            info['epoch_loss_list'] = []
            info['mode'] = 'inference'
            info['nth_epoch'] -= 1  # Redo the epoch as validation
        else:
            info['mode'] = 'training'
            # ins[0].switch_source(0)
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

    def __init__(self, root, mode='Training'):
        self.root = root
        self.mode = mode
        self.call_list = None
        self.valid_disp = None
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
        self.register_event('Teardown', _norm_finish)

    def prepare_execution(self):
        call_list, special_ops = self.root.get_executor_info()
        self.call_list = call_list
        self.dispatchers = special_ops['graph_inputs']
        self.loss = special_ops['losses']
        self.root_op = special_ops['root_op']

    def prepare_validation(self):
        call_list, special_ops = self.root.get_executor_info()
        self.validation_list = call_list
        self.valid_disp = special_ops['graph_inputs']
        self.valid_loss = special_ops['losses']
        self._set_validation()

    def execute(self, feed_dict=None, validation_feed_dict=None, epochs=1, progress=True):
        '''
          This function executes computational graph.

          Args:
              epochs (int): Number of epochs.
              progress (bool): If True is given, the progress will be shown.
        '''

        if validation_feed_dict is not None:
            for key, value in validation_feed_dict.items():
                self.root.feed(key, value)
            self.prepare_validation()
        if feed_dict is not None:
            for key, value in feed_dict.items():
                self.root.feed(key, value)

        if self.call_list is None or feed_dict is not None:
            self.prepare_execution()

        exe_info = {'inputs': self.dispatchers,
                    'losses': self.loss,
                    'progress': progress,
                    'mode': self.mode,
                    }

        if len(self.dispatchers) == 0 or \
                not any('input' in d.roles for d in self.dispatchers):
            warnings.warn('Trying to run executor without any dispatchers!\n' +
                          'Make sure that there is a valid dispatcher before executing.')
            raise NotImplementedError('Currently static input is not supported')

        self.print_info()

        for ev in self._events['Initialize']:
            ev(exe_info)

        # while exe_info['nth_epoch'] < epochs:
        for e in range(epochs):
            self.perform_event_epoch(exe_info)
            if validation_feed_dict is not None:
                with self.validation_mode(exe_info):
                    self.perform_event_epoch(exe_info)
        for ev in self._events['Teardown']:
            ev(exe_info)

    @contextmanager
    def validation_mode(self, exe_info):
        tmp1 = self.dispatchers
        tmp2 = self.call_list
        self.dispatchers = self.valid_disp
        self.call_list = self.validation_list
        exe_info['inputs'] = self.dispatchers
        yield
        self.dispatchers = tmp1
        self.call_list = tmp2
        exe_info['inputs'] = self.dispatchers

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
                    if mode == 'step' or mode == 'inference':
                        orig_inf = call._inference
                        call._inference = True
                    if rm.logging_level >= 10:
                        call.logged_perform()
                    else:
                        call.perform()
                    if mode == 'step' or mode == 'inference':
                        call._inference = orig_inf

        for ev in self._events['Step-Finish']:
            ev(exe_info)

    def register_event(self, event_name, event_function):
        assert isinstance(event_name, str) and event_name in self._events
        assert callable(event_function)
        self._events[event_name].append(event_function)

    def unregister_events(self, event_name):
        assert isinstance(event_name, str) and event_name in self._events
        self._events[event_name] = []

    def _set_validation(self):
        self.register_event('Epoch-Finish', _validation_func())

    def print_info(self):
        if len(self.dispatchers) >= 1 and \
           isinstance(self.dispatchers[0], rm.graph.distribution.put_graph.put_op):
            dis = self.dispatchers[0]
            if self.valid_disp is not None:
                have_validation = True
                v_d_num = len(self.valid_disp[0])
            else:
                have_validation = False
                v_d_num = 0
            t_d_num = len(dis)
            tot_num = t_d_num + v_d_num
            if self.mode == 'inference':
                m_depth = max(self.call_list['Forward'].keys())
            else:
                update_calls = list(self.call_list['Gradient'].keys())
                backward_calls = list(self.call_list['Backward'].keys())
                m_depth = max(update_calls + backward_calls)
            total = 0
            for part in self.call_list:
                for depth in self.call_list[part]:
                    total += len(self.call_list[part][depth])
            print('Train Data num: {0:>6d} ({1:3.0%})'.format(t_d_num, t_d_num / tot_num))
            if have_validation is True:
                print('Valid Data num: {0:>6d} ({1:3.0%})'.format(v_d_num, v_d_num / tot_num))
            print('Graph max depth is:', m_depth)
            print('Total number of nodes executed is:', total)
            print('Mode:', self.mode)

    def step(self, feed_dict):
        if feed_dict is not None:
            for key, value in feed_dict.items():
                self.root.feed(key, value)
        self.prepare_execution()
        exe_info = {
            'mode': 'step',
            'losses': self.loss,
        }
        self.perform_event_step(exe_info)
        loss = self.root_op.as_ndarray()
        return loss
