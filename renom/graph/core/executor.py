import numpy as np
from tqdm import tqdm
import renom as rm
import warnings


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
            info['step_loss'] = float(loss[0].as_ndarray())
        return
    epoch_loss_list = info['epoch_loss_list']
    bar = info['bar']
    epoch_name = info['epoch_name']

    if len(loss) > 0:
        loss = float(loss[0].as_ndarray())
        bar.set_description("{0!s: >10} cur-loss={1:5.3f}".format(epoch_name, loss))
        epoch_loss_list.append(loss)
    else:
        bar.set_description("{0!s: >10}".format(epoch_name))
    bar.update(1)


def _norm_epoch_finish(info):
    epoch_loss_list = info['epoch_loss_list']
    bar = info['bar']
    all_losses = info['all_losses']
    epoch_name = info['epoch_name']
    loss = info['losses']

    epoch_loss_list.pop(-1)
    all_losses.append(np.sum(epoch_loss_list))
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
            ins[0].switch_source(1)
            info['epoch_loss_list'] = []
            info['mode'] = 'inference'
            info['nth_epoch'] -= 1  # Redo the epoch as validation
        else:
            info['mode'] = 'training'
            ins[0].switch_source(0)
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
        self.input_sources = {}
        for disp in self.dispatchers:
            if hasattr(disp, 'keyword') and disp.keyword is not None:
                self.input_sources[disp.keyword] = disp
        self.loss = special_ops['losses']
        self.root = special_ops['root']
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
        self.register_event('Teardown', _norm_finish)

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

        if len(self.dispatchers) == 0 or \
                not any('input' in d.roles for d in self.dispatchers):
            warnings.warn('Trying to run executor without any dispatchers!\n' +
                          'Make sure that there is a valid dispatcher before executing.')
            raise NotImplementedError('Currently static input is not supported')

        if len(self.dispatchers) >= 1 and \
           isinstance(self.dispatchers[0], rm.graph.utils.distributor.dispatch):
            dis = self.dispatchers[0]
            if len(dis._value_list) == 2:
                have_validation = True
                v_d_num = len(dis._value_list[1])
            else:
                have_validation = False
                v_d_num = 0
            t_d_num = len(dis._value_list[0])
            tot_num = t_d_num + v_d_num
            if self.mode == 'inference':
                m_depth = max(self.call_list['Forward'].keys())
            else:
                m_depth = max(self.call_list['Gradient'].keys())
            total = 0
            for part in self.call_list:
                for depth in self.call_list[part]:
                    total += len(self.call_list[part][depth])
            recognized = []
            for disp in self.dispatchers:
                if disp.keyword is not None:
                    recognized.append(disp.keyword)
            print('Train Data num: {0:>6d} ({1:3.0%})'.format(t_d_num, t_d_num / tot_num))
            if have_validation is True:
                print('Valid Data num: {0:>6d} ({1:3.0%})'.format(v_d_num, v_d_num / tot_num))
            if len(recognized) > 0:
                print('Recognized input sources are:', *recognized)
            print('Graph max depth is:', m_depth)
            print('Total number of nodes executed is:', total)
            print('Mode:', self.mode)

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
        assert len(self.dispatchers) > 0 and \
            len(self.dispatchers[0]._value_list) > 1
        self.register_event('Epoch-Finish', _validation_func())

    def step(self, step_data=None):
        exe_info = {
            'mode': 'step',
            'losses': self.loss,
        }
        if step_data is not None:
            assert isinstance(step_data, dict)
            for key in step_data.keys():
                assert isinstance(step_data[key], np.ndarray)
                if key in self.input_sources:
                    self.input_sources[key].value = step_data[key]
        self.perform_event_step(exe_info)
        #loss = self.loss[0].as_ndarray()
        loss = self.root.as_ndarray()
        if step_data is not None:
            for dis in self.dispatchers:
                dis.switch_source(0)
        return loss

    def set_input_data(self, new_data):
        assert isinstance(new_data, dict)
        self.unregister_events('Epoch-Finish')
        self.register_event('Epoch-Finish', _norm_epoch_finish)
        for key in new_data:
            if key in self.input_sources:
                if not isinstance(self.input_sources[key], rm.graph.utils.distributor.dispatch):
                    raise NotImplementedError(
                        'Currently can only set input sources for dispatchers.')
                self.input_sources[key].change_input(new_data[key])
