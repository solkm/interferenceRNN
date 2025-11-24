from .psychrnn.tasks.task import Task
import numpy as np

class BaseHistoryTask(Task):
    # Task timing constants (in ms)
    DELAY1_DUR = 500
    STIM_DUR = 200
    DELAY3_DUR = 150
    DELAY2_RANGE = (300, 500)
    CHOICE_TOTAL = 750 # total time for delay2 + choice period
    T = DELAY1_DUR + 2*STIM_DUR + CHOICE_TOTAL + DELAY3_DUR

    def __init__(self, dat, dat_inds, K=10, test_all=False, 
                 vis_noise=0.8, mem_noise=0.5, fix_noise=0.2,
                 fixed_delay2=None, fixed_sl=None, fixed_sf=None,
                 dt=10, tau=100, N_batch=100, **kwargs):
        
        self.dat = dat
        self.dat_inds = dat_inds
        self.K = K
        self.N_hist_in = 6 * (K - 1) # dimension of history inputs
        self.test_all = test_all
        self.vis_noise = vis_noise
        self.mem_noise = mem_noise
        self.fix_noise = fix_noise
        self.fixed_delay2 = fixed_delay2
        self.fixed_sl = fixed_sl
        self.fixed_sf = fixed_sf
        
        mstim = kwargs.get('mstim', False)
        N_in = self.N_hist_in + 3 + int(mstim)
        N_out = 7
        N_batch = dat_inds.shape[0] if test_all else N_batch
        
        if dat_inds is not None:
            self._check_dat_inds_validity()
        super().__init__(N_in, N_out, dt, tau, self.T, N_batch)

    def _check_dat_inds_validity(self):
        # Make sure dat_inds are valid for the given K
        K_trainable = np.sort([
            int(col.split('K')[1].split('trainable')[0])
            for col in self.dat.columns
            if col.startswith('K') and col.endswith('trainable')
        ])
        K_closest = K_trainable[K_trainable >= self.K][0]
        assert np.all(self.dat.loc[self.dat_inds, f'K{K_closest}trainable'] == 1), \
            f'dat_inds must be at least K{self.K} trainable'

    def _get_trial_stimuli(self, change, fixed_feature=None):
        """
        Get the values of a feature (sl/sf) for the two stimuli in a trial, 
            handling fixed values.
        Args:
            change (float): The difference between the two stimuli.
            fixed_feature (tuple or None): If not None, a tuple (f1, f2) where 
                f1, f2 can be fixed values (float) or None (not fixed).
        Returns:
            f1 (float): Value of the first stimulus.
            f2 (float): Value of the second stimulus.
            change (float): The difference between the two stimuli. May override
                the input change if both stimuli are fixed.
        """
        if not fixed_feature or not any(fixed_feature):
            f1 = np.random.uniform(2, 3)
            return f1, f1 + change, change
    
        f1_fixed, f2_fixed = fixed_feature
        
        if f1_fixed is not None and f2_fixed is not None: # both stimuli fixed
            return f1_fixed, f2_fixed, f2_fixed - f1_fixed
        
        elif f1_fixed is not None: # only first stimulus fixed
            return f1_fixed, f1_fixed + change, change
        
        elif f2_fixed is not None: # only second stimulus fixed
            return f2_fixed - change, f2_fixed, change
    
    def generate_trial_params(self, batch, trial):
        """" 
        Define parameters for each trial.

        Args:
            batch (int): The batch number that this trial is part of.
            trial (int): The trial number of the trial within the batch.
    
        Returns:
            dict: Dictionary of trial parameters.
        """

        # Select trial index, extract trial history data
        i = self.dat_inds[trial] if self.test_all else np.random.choice(self.dat_inds)
        hist_slice = slice(i - self.K + 1, i + 1)
        
        params = {
            'trial_ind': i,
            'choice': np.array(self.dat['choice'][hist_slice], dtype='int'),
            'correct': np.array(self.dat['correct'][hist_slice], dtype='int'),
            'dsf': np.array(self.dat['dsf'][hist_slice]) / 2,
            'dsl': np.array(self.dat['dsl'][hist_slice]) / 2,
            'task': np.array(self.dat['task'][hist_slice], dtype='int'),
            'm_task': np.array(self.dat['m_task'][hist_slice], dtype='int')
        }

        # Define current trial stimuli
        params['sf1'], params['sf2'], params['dsf'][-1] = \
            self._get_trial_stimuli(params['dsf'][-1], self.fixed_sf)
        
        params['sl1'], params['sl2'], params['dsl'][-1] = \
            self._get_trial_stimuli(params['dsl'][-1], self.fixed_sl)

        params['delay2_dur'] = (self.fixed_delay2 if self.fixed_delay2 is not None
                                else np.random.uniform(*self.DELAY2_RANGE))
         
        return params

    def _init_inputs_with_noise(self):
        """ 
        Initialize the trial inputs with noise.
        """
        x_t = np.sqrt(2) * np.random.standard_normal(self.N_in)

        # Scale noise for each input type:
        N_hist_in = self.N_hist_in
        x_t[:N_hist_in] *= self.mem_noise # trial history inputs
        x_t[N_hist_in : N_hist_in + 2] *= self.vis_noise # stimulus inputs
        x_t[N_hist_in + 2] *= self.fix_noise # fixation input

        return x_t
    
    def _get_history_inputs(self, params):
        hist_ins = np.zeros(6 * (self.K - 1))
        choice = params['choice']
        correct = params['correct']
        dsl = params['dsl']
        dsf = params['dsf']

        for i in range(self.K - 1):
            if choice[i] != 0:
                # Change amount of the chosen task feature (perceptual difficulty)
                hist_ins[6 * i] = 0.2 + (dsl[i] if choice[i] >= 3 else dsf[i])

                # 1-hot encoded choice
                hist_ins[6 * i + choice[i]] = 1

                # Reward feedback
                hist_ins[6 * i + 5] = 1 if choice[i] == correct[i] else -1

        return hist_ins
    
    def _get_stimulus_inputs(self, time, params):
        stim_ins = np.zeros(2)
        stim1_on = self.DELAY1_DUR
        stim_dur = self.STIM_DUR
        stim2_on = stim1_on + stim_dur + params['delay2_dur']
        
        if stim1_on < time < stim1_on + stim_dur:
            stim_ins[0] = params['sf1']
            stim_ins[1] = params['sl1']

        elif stim2_on < time < stim2_on + stim_dur:
            stim_ins[0] = params['sf2']
            stim_ins[1] = params['sl2']

        return stim_ins
    
    def _get_target_task_outputs(self, params):
        task_outs = np.zeros(2)
        m_task = params['m_task'][-1] # coded as 1 or 2
        task_outs[m_task - 1] = 1
        return task_outs

    def _get_target_choice_outputs(self, params):
        choice_outs = np.zeros(4)
        choice = params['choice'][-1] # coded as 1-4
        choice_outs[choice - 1] = 1
        return choice_outs

    def _add_custom_inputs(self, x_t, time, params):
        """
        Placeholder for any additional custom input modifications.
        """
        return x_t

    def _scale_mask(self, mask_t, params):
        """
        Placeholder for any custom mask scaling based on params.
        """
        return mask_t
    
    def trial_function(self, time, params): 
        """ 
        Based on the params compute the trial stimulus (x_t), correct output (y_t), 
        and mask (mask_t) at the given time.
    
        Args:
            time (int): The time within the trial (0 <= time < T).
            params (dict): The trial params produced generate_trial_params()
    
        Returns:
            x_t (ndarray(dtype=float, shape=(N_in,))): Trial input at time given params.
            y_t (ndarray(dtype=float, shape=(N_out,))): Correct trial output at time given params.
            mask_t (ndarray(dtype=int, shape=(N_out,))): Output mask for training.
        """
        N_hist_in = self.N_hist_in

        # Initialize inputs, target outputs, and output mask
        x_t = self._init_inputs_with_noise()
        y_t = np.zeros(self.N_out)
        mask_t = np.ones(self.N_out)

        # Add trial history inputs (time-independent)
        x_t[:N_hist_in] += self._get_history_inputs(params)

        # Go time
        t_go = self.DELAY1_DUR + 2 * self.STIM_DUR + params['delay2_dur'] + self.DELAY3_DUR

        if time > 100: # target task outputs
            y_t[:2] = self._get_target_task_outputs(params)
            mask_t[:2] *= 2

        if time < t_go:
            # Current trial stimuli
            x_t[N_hist_in : N_hist_in + 2] += self._get_stimulus_inputs(time, params)
            x_t[N_hist_in + 2] += 1 # fixation input
            y_t[-1] = 1 # fixation output
        else:
            mask_t[2:6] *= 4
            y_t[2:6] = self._get_target_choice_outputs(params)

        # Optionally add additional custom inputs
        x_t = self._add_custom_inputs(x_t, time, params)

        # Optionally scale mask based on params
        mask_t = self._scale_mask(mask_t, params)
        
        return x_t, y_t, mask_t
    
class MonkeyHistoryTask(BaseHistoryTask):
    def __init__(self, randomized_errs=False, mstim=False, 
                 mstim_noise=0.2, mstim_strength=1, **kwargs):
        
        self.randomized_errs = randomized_errs
        self.mstim = mstim
        if mstim:
            self.mstim_noise = mstim_noise
            self.mstim_strength = mstim_strength
    
        super().__init__(mstim=mstim, **kwargs)
    
    def _add_custom_inputs(self, x_t, time, params):
        """
        Handle optional microstimulation input.
        """
        if self.mstim:
            stim1_on = self.DELAY1_DUR
            stim_dur = self.STIM_DUR
            stim2_on = stim1_on + stim_dur + params['delay2_dur']

            # Scale noise
            x_t[-1] *= self.mstim_noise

            # Add microstimulation during stimulus 2
            if stim2_on < time < stim2_on + stim_dur:
                x_t[-1] += self.mstim_strength
        return x_t
    
    def _get_target_choice_outputs(self, params):
        if self.randomized_errs and choice != correct:
            choice_outs = np.zeros(4)
            correct = params['correct'][-1]
            choice = params['choice'][-1]
            rand_err = np.random.choice(np.delete(np.arange(1, 5), correct - 1))
            choice_outs[rand_err - 1] = 1
        else:
            choice_outs = super()._get_target_choice_outputs(params)
        return choice_outs

    def _scale_mask(self, mask_t, params):
        """
        Scale mask to emphasize switch trials.
        """
        if params['m_task'][-2] != params['m_task'][-1]:
            mask_t[:6] *= 1.5
        return mask_t

class SelfHistoryTask(BaseHistoryTask):
    def __init__(self, targ_correct=True, **kwargs):

        super().__init__(**kwargs)
        self.start_inds = np.array([
            i for i in self.dat_inds if i - 1 not in self.dat_inds]) - (self.K - 1)
        self.end_inds = np.array([i for i in self.dat_inds if i + 1 not in self.dat_inds])
        self.targ_correct = targ_correct

    def train_batch_generator(self, hist):
        """ 
        Generates a batch of trials for training.
        This trains the network using its OWN trial history (stored in 'hist').
        To see how 'hist' is initialized and updated from model outputs during 
            training, see self_history.RNN_SH2.train.
        """
        K = self.K
        dat = self.dat
        N_batch = self.N_batch
        start_inds = self.start_inds
        end_inds = self.end_inds

        x_data = [] # inputs
        y_data = [] # target outputs
        mask = [] # output masks
        
        # Extend trial history arrays for the upcoming batch
        for key in ['choice', 'correct', 'dsl', 'dsf', 'task', 't_ind', 't_sess']:
            hist[key] = np.append(hist[key], np.zeros((N_batch, 1)), axis=1)
        
        for trial in range(self.N_batch):
            prev_t_sess = hist['t_sess'][trial, -2] # t_sess is trial-in-session
            prev_t_ind = hist['t_ind'][trial, -2] # t_ind is trial index in data

            if (prev_t_sess == 0) or np.isin(prev_t_ind, end_inds): # if it's the first epoch or the previous trial was an end_ind
                t_ind = np.random.choice(start_inds) # choose a new start ind
                t_sess = 1 # reset t_sess
            else: # otherwise continue
                t_ind = prev_t_ind + 1
                t_sess = prev_t_sess + 1

            hist['t_ind'][trial, -1] = t_ind
            hist['t_sess'][trial, -1] = t_sess

            trial_data = dat.loc[t_ind]
            hist['correct'][trial, -1] = trial_data['correct']
            hist['task'][trial, -1] = trial_data['task']
            hist['dsl'][trial, -1] = trial_data['dsl'] / 2
            hist['dsf'][trial, -1] = trial_data['dsf'] / 2
            
            params = {
                'choice': np.array(hist['choice'][trial, -K:].copy(), dtype='int'),
                'correct': np.array(hist['correct'][trial, -K:].copy(), dtype='int'),
                'task': np.array(hist['task'][trial, -K:].copy(), dtype='int'),
                'dsl': hist['dsl'][trial, -K:].copy(),
                'dsf': hist['dsf'][trial, -K:].copy(),
                'delay2_dur': np.random.uniform(300, 500), # variable delay period
                'm_choice': int(trial_data['choice']), # only used if not targ_correct
                'm_task': int(trial_data['task']) # only used if not targ_correct
            }
            
            if t_sess < K: # if at beginning of session, zero out unavailable history
                for key in ['choice', 'correct', 'dsl', 'dsf', 'task']:
                    params[key][:-int(t_sess)] = 0

            params['sf1'] = np.random.uniform(2, 3)
            params['sf2'] = params['sf1'] + params['dsf'][-1]
            params['sl1'] = np.random.uniform(2, 3)
            params['sl2'] = params['sl1'] + params['dsl'][-1]
            
            x, y, m = self.generate_trial(params)
            x_data.append(x)
            y_data.append(y)
            mask.append(m)
            
        return np.array(x_data), np.array(y_data), np.array(mask), hist

    def _get_target_task_outputs(self, params):
        task_outs = np.zeros(2)
        if self.targ_correct:
            targ_task = params['task'][-1] # correct task
        else:
            targ_task = params['m_task'] # monkey task
        task_outs[targ_task - 1] = 1
        return task_outs
    
    def _get_target_choice_outputs(self, params):
        choice_outs = np.zeros(4)
        if self.targ_correct:
            targ_choice = params['correct'][-1] # correct choice
        else:
            targ_choice = params['m_choice'] # monkey choice
        choice_outs[targ_choice - 1] = 1
        return choice_outs
