"""
Functions to generate signals that resemble MT responses to random dot motion stimuli.
These are cartoonified, with a ring of N rate-coded neurons tuned to different directions.

Currently, the population is uniform, but future versions might include a range of tuning curves
"""

import numpy as np
import numpy.typing as npt
import copy
from typing import Callable

class temporal_tuning:
    def __init__(self, baseline : float = 0.0,
                       rise_scale : float = 1.0, sustain_scale : float = 0.5,
                       rise_timescale : float = 2.0,   sustain_timescale : float = 10.0,
                       off_rise_scale : float = 0,
                       off_rise_timescale : float = 2.0, off_decay_timescale : float = 2.0):
        self.rise_scale_     = rise_scale
        self.sustain_scale_  = sustain_scale
        self.off_rise_scale_ = off_rise_scale
        self.sustain_scale_  = sustain_scale

        self.rise_timescale_  = rise_timescale
        self.sustain_timescale_  = sustain_timescale
        self.off_rise_timescale_  = off_rise_timescale
        self.off_decay_timescale_ = off_decay_timescale
        self.baseline_ = baseline

    def generate_response(self,max_time : int = 55, off_time : int = 50,  gain : float = 1.0,
                            return_parts : bool = False) -> npt.NDArray[np.float_]:
        tt = np.arange(max_time)
        tt2 = tt - off_time

        rise  = self.rise_scale_*(1.0 - np.exp(-tt/self.rise_timescale_))
        decay = (self.sustain_scale_-self.rise_scale_)*(1.0 - np.exp(-tt/self.sustain_timescale_))

        off_rise  = (  self.off_rise_scale_-self.sustain_scale_)*(1.0 - np.exp(-tt2/self.off_rise_timescale_))
        off_decay = (0-self.off_rise_scale_)*(1.0 - np.exp(-tt2/self.off_decay_timescale_))
        off_rise[ tt2 < 0] = 0;
        off_decay[tt2 < 0] = 0;

        if(return_parts):
            return (gain*rise  + self.baseline_,gain*decay,gain*off_rise,gain*off_decay,tt)
        else:
            return gain*(rise+decay+off_rise+off_decay)  + self.baseline_
        

DEFAULT_DC   = temporal_tuning(rise_scale=1.0, sustain_scale= 0.5,
                       rise_timescale=2.0,   sustain_timescale=10.0,
                       off_rise_scale=0,
                       off_rise_timescale=2.0, off_decay_timescale=2.0);

DEFAULT_STIM = temporal_tuning(rise_scale=0.75, sustain_scale= 0.25,
                       rise_timescale=2.0,   sustain_timescale=10.0,
                       off_rise_scale=0,
                       off_rise_timescale=2.0, off_decay_timescale=2.0);


class motion_tuned_unit:

    def __init__(self, dc_response : temporal_tuning = DEFAULT_DC, stim_response : temporal_tuning = DEFAULT_STIM,
                       noise_response : temporal_tuning = None, noise_kernel : Callable[[int],npt.NDArray] = np.eye,
                       tuning_curve : Callable[[npt.ArrayLike],npt.NDArray] = np.cos):
        self.dc_response_ : temporal_tuning   = copy(dc_response)
        self.stim_response_ : temporal_tuning = copy(stim_response)

        self.baseline_ = self.stim_response_.baseline_ + self.stim_response_.baseline_
        self.dc_response_.baseline_ = 0
        self.stim_response_.baseline_ = 0

        if(noise_response is None):
            mm = 0.5
            bb = np.max([self.dc_response_.rise_scale_, self.dc_response_.off_rise_scale_])
            self.noise_response_  = temporal_tuning(baseline = bb, rise_scale=-mm*self.dc_response_.rise_scale_, sustain_scale=-mm*self.dc_response_.sustain_scale_,
                       rise_timescale=self.dc_response_.rise_timescale_,   sustain_timescale=self.dc_response_.sustain_timescalee_,
                       off_rise_scale=-mm*self.dc_response_.off_rise_scale_,
                       off_rise_timescale=self.dc_response_.off_rise_timescale_, off_decay_timescale=self.dc_response_.off_decay_timescale_);
        else:
            self.noise_response_   = copy(noise_response)

        self.tuning_curve_ = tuning_curve
        self.noise_kernel_ = noise_kernel
        self.noise_sigma_ = None

    
    def noise_sigma(self, size : int) -> npt.NDArray:
        if(self.noise_sigma_ is None or self.noise_sigma_.shape[0] < size):
            self.noise_sigma_ = self.noise_kernel_(size)
        return self.noise_sigma_[:size,:size]

    def generate_response(self, onset_offset_times : list[tuple[int,int]],
                                stimulus : npt.ArrayLike,
                                coherence_gain : npt.ArrayLike,
                                contrast_gain : npt.ArrayLike,
                                return_parts : bool = False,
                                rng : np.random.Generator = None):
        Z = self.tuning_curve_(stimulus) * coherence_gain * contrast_gain # stimulus

        X = np.zeros_like(stimulus).ravel() + self.baseline_ # DC response, modulated by contrast_gain
        Y = np.zeros_like(stimulus).ravel() # stimulus response, modulated by contrast_gain * coherence_gain * tuning_curve(stimulus)
        sig = np.zeros_like(stimulus).ravel() + self.noise_response_.baseline_

        T = X.size

        for onset, offset in onset_offset_times:
            X[onset:]   += self.dc_response_.generate_response(T-onset, off_time=offset-onset,  gain=contrast_gain[onset:])
            Y[onset:]   += self.stim_response_.generate_response(T-onset, off_time=offset-onset,  gain=Z[onset:])
            sig[onset:] += self.noise_response_.generate_response(T-onset, off_time=offset-onset) - self.noise_response_.baseline_

        if(return_parts):
            return X,Y,sig
        else:
            if(rng is None):
                return X+Y
            else:
                noise = rng.multivariate_normal(np.zeros(T), self.noise_sigma(T)) * sig
                return X + Y + noise

class dmc_stimulus_generator:
    def __init__(self, sample_time : int | list = 5,
                       sample_length : int | list = 25, test_length : int | list = 25,
                       delay_length : int | list = 50, post_length : int | list = 5,
                       coherence_orientation_noise_scale : Callable[[float],float] = lambda cc : 0,
                       coherence_gain_noise_scale : Callable[[float],float] = lambda cc : 0,
                       coherence_noise_temporal_kernel : Callable[[int],npt.NDArray] = np.eye) :
        self.sample_time_ = sample_time
        self.sample_length_ = sample_length
        self.test_length_ = test_length
        self.delay_length_ = delay_length
        self.post_length_  = post_length

        self.coherence_orientation_noise_scale_ = coherence_orientation_noise_scale
        self.coherence_gain_noise_scale_ = coherence_gain_noise_scale
        self.coherence_noise_temporal_kernel_ = coherence_noise_temporal_kernel

    def coherence_noise_sigma(self, size : int) -> npt.NDArray:
        if(self.coherence_noise_sigma_ is None or self.coherence_noise_sigma_.shape[0] < size):
            self.coherence_noise_sigma_ = self.coherence_noise_temporal_kernel_(size)
        return self.coherence_noise_sigma_[:size,:size]
         
    
    def generate_trial(self, 
                       sample_orientation : float,
                       test_orientation : float,
                       is_match : bool,
                       sample_contrast : float = 1.0, sample_coherence : float = 1.0,
                       test_contrast : float = 1.0, test_coherence : float = 1.0,
                       rng : np.random.Generator = None):
        if(np.isscalar(self.sample_time_)):
            sample_time = self.sample_time_
        else:
            sample_time = rng.choice( self.sample_time_)

        if(np.isscalar(self.sample_length_)):
            sample_length = self.sample_length_
        else:
            sample_length = rng.choice( self.sample_length_)

        if(np.isscalar(self.test_length_)):
            test_length = self.test_length_
        else:
            test_length = rng.choice( self.test_length_)

        if(np.isscalar(self.delay_length_)):
            delay_length = self.delay_length_
        else:
            delay_length = rng.choice( self.delay_length_)
            
        if(np.isscalar(self.post_length_)):
            post_length = self.post_length_
        else:
            post_length = rng.choice( self.post_length_)

        trial_length = sample_time + sample_length + delay_length + test_length + post_length

        test_time = sample_time + sample_length + delay_length

        stims =  [(sample_time, sample_time+sample_length,sample_orientation,sample_coherence,sample_contrast),
                  (test_time, test_time+test_length,test_orientation,test_coherence,test_contrast)]
        onset_offset_times =  [(sample_time, sample_time+sample_length),
                  (test_time, test_time+test_length)]

        stimulus = np.zeros(trial_length)
        contrast_gain = np.zeros(trial_length)
        coherence_gain = np.zeros(trial_length)
        
        for onset, offset, orientation, coherence, contrast in stims:
            T = offset-onset

            if(rng is not None):
                noise = rng.multivariate_normal(np.zeros(T), self.coherence_noise_sigma(T), size=2)
                coherence_noise = self.coherence_gain_noise_scale_(coherence)*noise[0,:]
                orientation_noise = self.coherence_orientation_noise_scale_(coherence)*noise[1,:]
            else:
                coherence_noise = 0
                orientation_noise = 0

            contrast_gain[onset:offset] = contrast
            coherence_gain[onset:offset] = coherence + coherence_noise
            stimulus[onset:offset] = orientation + orientation_noise

        output = np.zeros((trial_length,3))
        output[test_time:,2] = 1
        output[:test_time,:2] = np.nan
        if(is_match):
            output[test_time:,0] = 1
        else:
            output[test_time:,1] = 1

        return (onset_offset_times, stimulus, coherence_gain, contrast_gain, output)
    
    class dmc_population:
        def __init__(self, n_units : int = 12):

            self.units : list[motion_tuned_unit]= [];
            self.preferred_directions = np.linspace(0,2*np.pi, n_units, endpoint=False)
            for dd in self.preferred_directions:
                tc = lambda theta : np.cos(theta - dd)
                unit = motion_tuned_unit(tuning_curve = tc)
                self.units.append(unit)

            self.stimulus_generator = dmc_stimulus_generator()

        def generate_trial(self, 
                       sample_orientation : float,
                       test_orientation : float,
                       is_match : bool,
                       sample_contrast : float = 1.0, sample_coherence : float = 1.0,
                       test_contrast : float = 1.0, test_coherence : float = 1.0,
                       rng : np.random.Generator = None, return_all : bool = False):
            onset_offset_times, stimulus, coherence_gain, contrast_gain, output = self.stimulus_generator(sample_orientation=sample_orientation,
                                                                                                          test_orientation=test_orientation, is_match=is_match,
                                                                                                          sample_contrast=sample_contrast, sample_coherence=sample_coherence,
                                                                                                          test_contrast=test_contrast,test_coherence=test_coherence,
                                                                                                          rng=rng)
            X = np.zeros((stimulus.size, len(self.units)))
            for ii, unit in enumerate(self.units):
                X[:,ii] = unit.generate_response(onset_offset_times=onset_offset_times, stimulus=stimulus,
                                       contrast_gain=contrast_gain, coherence_gain=coherence_gain, rng=rng)
            if(return_all):
                return (X, output)
            else:
                return (X, output, onset_offset_times, stimulus, coherence_gain, contrast_gain)
            

        def generate_trials(self, N_match : int, N_non_match : int, rng : np.random.Generator):
            trs = []
            for ii in (N_match + N_non_match):
                dd = np.pi/16
                s1 = rng.uniform(0,np.pi - dd,size=(2)) + rng.uniform(0,dd,size=(2))
                if(ii > N_match):
                    s1 += np.array([0,np.pi])
                    match = False
                else:
                    match = True
                if(rng.binomial(1,0.5) == 0):
                    s1 += np.pi
                trs.append((s1, match))

            X = []
            output = []
            for orientations, is_match in orientations:
                X_c, output_c = self.generate_trial(orientations[0], orientations[1], is_match)
                X.append(X_c)
                output.append(output_c)

            X = np.dstack(X)
            output = np.dstack(output)
            return (X,output)