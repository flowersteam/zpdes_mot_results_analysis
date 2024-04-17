import numpy as np
import pandas as pd

# usage
# class_results = Results_memory(pandas_data_frame,index_of_observer)
# The pandas data frame is the one loaded from the csv result file of the memorability task.
# I assume that the dataframe is separated for each observer.

# output
# You can extract condition-wise results using this class.
# print(np.mean(class_results.out_mat_hit_miss,1))
# print(np.mean(class_results.out_mat_fa_cr,1))
# print(class_results.rt_all_mean)
# print(class_results.rt_correct_mean)


# make the condition sorting
class Results_memory():
    def __init__(self, df, num_stimcond=5, num_repetition=8):
        self.num_stimcond = num_stimcond
        self.num_repetition = num_repetition

        ##results arrays
        self.out_mat_hit_miss = []
        self.out_mat_fa_cr = []
        self.out_mat_rt_cond = []
        self.out_mat_rt_cond_std = []
        self.rt_all = []
        self.rt_correct = []

        self.df = df
        self.ind_labels = [2, 3, 4, 5, 99]  # 2,3,4,5,>99

        self.cal_allrepetitions()

    def cal_allrepetitions(self):
        for t in range(len(self.df)):
            out_mat_hit_miss, out_mat_fa_cr, rt_all, rt_correct, out_mat_rt_cond = self.cal_separate_eachcond(t)
            if t == 0:
                self.out_mat_hit_miss = out_mat_hit_miss
                self.out_mat_fa_cr = out_mat_fa_cr
                self.rt_all = rt_all
                self.rt_correct = rt_correct
                self.out_mat_rt_cond = out_mat_rt_cond
            else:
                self.out_mat_hit_miss = np.concatenate((self.out_mat_hit_miss, out_mat_hit_miss), axis=1)
                self.out_mat_fa_cr = np.concatenate((self.out_mat_fa_cr, out_mat_fa_cr), axis=1)
                self.out_mat_rt_cond = np.concatenate((self.out_mat_rt_cond, out_mat_rt_cond), axis=1)
                self.rt_all = np.concatenate((self.rt_all, rt_all), axis=0)
                self.rt_correct = np.concatenate((self.rt_correct, rt_correct), axis=0)

        self.out_mat_hit_miss_mean = np.mean(self.out_mat_hit_miss, 1)
        self.out_mat_hit_miss_sum = np.sum(self.out_mat_hit_miss, 1)
        self.out_mat_fa_cr_mean = np.mean(self.out_mat_fa_cr, 1)
        self.out_mat_fa_cr_sum = (len(self.df) * self.num_repetition) - np.sum(self.out_mat_fa_cr, 1)

        tmp = [np.mean(self.out_mat_rt_cond[t][np.where(self.out_mat_rt_cond[t] != 10000.)]) for t in
               range(self.num_stimcond)]
        tmp = [1400. if np.isnan(tmp[t]) else tmp[t] for t in range(self.num_stimcond)]
        tmp_std = [np.std(self.out_mat_rt_cond[t][np.where(self.out_mat_rt_cond[t] != 10000.)]) for t in
                   range(self.num_stimcond)]
        tmp_std = [1400. if np.isnan(tmp_std[t]) else tmp_std[t] for t in range(self.num_stimcond)]
        self.out_mat_rt_cond = np.array(tmp)
        self.out_mat_rt_cond_std = np.array(tmp_std)

        self.rt_all_mean = np.mean(self.rt_all)
        self.rt_correct_mean = np.mean(self.rt_correct)

    def cal_separate_eachcond(self, ind_rep):
        out_mat_hit_miss = np.zeros(
            (self.num_stimcond, self.num_repetition))  # 5 condition and 8 reptation for each memorability task
        out_mat_fa_cr = np.zeros(
            (self.num_stimcond, self.num_repetition))  # 5 condition and 8 reptation for each memorability task

        # ind_targets = self.df.loc[ind_rep,'results_targetvalue']
        ind_targets = self.df['results_targetvalue'].iloc[ind_rep]
        # ind_targets = self.df.iloc[ind_rep,6]
        ind_targets = ind_targets.split(',')
        ind_targets = np.array(pd.to_numeric(ind_targets))

        # ind_stimconds = self.df.loc[ind_rep,'results_stimind']
        # ind_stimconds = self.df.iloc[ind_rep,8]
        ind_stimconds = self.df['results_stimind'].iloc[ind_rep]
        ind_stimconds = ind_stimconds.split(',')
        ind_stimconds = np.array(pd.to_numeric(ind_stimconds))

        # res_correct_obs = self.df.loc[ind_rep,'results_flagcorrect']
        # res_correct_obs = self.df.iloc[ind_rep,7]
        res_correct_obs = self.df['results_flagcorrect'].iloc[ind_rep]
        res_correct_obs = np.array(res_correct_obs.split(','))

        # for rt
        ind_rt = self.df.iloc[ind_rep, 5]
        ind_rt = self.df['results_rt'].iloc[ind_rep]
        ind_rt = ind_rt.split(',')
        tmp = [float(ind_rt[i] or "0") for i in range(len(ind_rt))]
        ind_rt = np.array(tmp)

        # make indices correponding to the first and second presetnations.
        first_stim = []
        first_ind = []
        first_response = []
        second_stim = []
        second_ind = []
        second_response = []

        for t in range(ind_targets.shape[0]):
            if ind_targets[t] == 1:
                first_stim.append(ind_stimconds[t])
                first_ind.append(t)
                first_response.append(float(res_correct_obs[t] == 'true'))
            elif ind_targets[t] == 2:
                second_stim.append(ind_stimconds[t])
                second_ind.append(t)
                second_response.append(float(res_correct_obs[t] == 'true'))
        second_stim = np.array(second_stim)
        first_stim = np.array(first_stim)

        # accuracy coupled with the stimulus conditions.
        ind_cond = []
        hist_miss = []
        fa_cr = []
        rt_condwise = []
        for t in range(len(first_stim)):
            tmp_ind = second_ind[int(np.where(second_stim == first_stim[t])[0])]
            ind_cond.append(tmp_ind - first_ind[t])
            hist_miss.append(float(res_correct_obs[tmp_ind] == 'true'))
            fa_cr.append(first_response[t])
            if res_correct_obs[tmp_ind] == 'true':
                rt_condwise.append(ind_rt[tmp_ind])
            else:
                rt_condwise.append(10000.)
        ind_cond = np.array(ind_cond)
        hist_miss = np.array(hist_miss)
        fa_cr = np.array(fa_cr)
        rt_condwise = np.array(rt_condwise)

        out_mat_rt_cond = np.zeros(
            (self.num_stimcond, self.num_repetition))  # 5 condition and 8 reptation for each memorability task
        # sort the results accouring to the results condition
        for t in range(out_mat_hit_miss.shape[0]):
            if t == out_mat_hit_miss.shape[0] - 1:
                out_mat_hit_miss[t, :] = hist_miss[np.where(ind_cond > self.ind_labels[t])]
                out_mat_fa_cr[t, :] = fa_cr[np.where(ind_cond > self.ind_labels[t])]
                out_mat_rt_cond[t, :] = rt_condwise[np.where(ind_cond > self.ind_labels[t])]
            else:
                out_mat_hit_miss[t, :] = hist_miss[np.where(ind_cond == self.ind_labels[t])]
                out_mat_fa_cr[t, :] = fa_cr[np.where(ind_cond == self.ind_labels[t])]
                out_mat_rt_cond[t, :] = rt_condwise[np.where(ind_cond == self.ind_labels[t])]

        rt_all = ind_rt[np.where(ind_rt != 0)]
        rt_all = np.array(rt_all)
        # self.rt_all_mean = np.mean(ind_rt[np.where(ind_rt!=0)])
        res_correct_obs[np.where(ind_targets != 2)] = 'false'
        rt_correct = ind_rt[np.where(res_correct_obs == 'true')]
        rt_correct = np.array(rt_correct)
        # self.rt_correct_mean = np.mean(ind_rt[np.where(res_correct_obs=='true')])

        return out_mat_hit_miss, out_mat_fa_cr, rt_all, rt_correct, out_mat_rt_cond
