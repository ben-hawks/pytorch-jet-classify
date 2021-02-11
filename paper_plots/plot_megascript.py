import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import ast
import re
import math
import yaml
import pandas as pd
import json
import pylab
import warnings
warnings.filterwarnings("ignore")
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
import mplhep as hep
hep.set_style(hep.style.ROOT)

# this plot script is a bonafide thermonuclear garbage fire, apologies to anyone passing through

# At some point a cleanup would be nice, but this was made to be able to
# get the job done at the time, and once the job was done not much needs to be
# looked at again. And it damn well does get the job (at leat the parts we care about) done,
# it's just damn ugly in the process

# This script takes and uses the .json files output by
# post_training_metrics.py and bo_post_training_metrics.py and plots all the results we
# think we might care about for them. The main function is randwise_plots, which will go through
# a collection of data files (each file being one model set (as defined by <precisions> below) at one randomization)
# and plot Accuracy, AUC, Neural Efficiency, and Bkgd Eff @ 50% signal Eff vs bops for each precision/rand %

# There's also some choice plots that are done by "hand" that were made for the purpose of the paper,
# typically plotting 32b vs 6b, or LT Pruning vs FT Pruning, along with some plots for each
# BO run (by precision).

# As a final note, gen_mean_err_set is intended to take a folder full of multiple instances of the
# previously mentioned data files, specifying a 'template' path to fill in with the rand values
# and range of 1 thru <kfolds> defined below (in that order, again, this was a "get it done" script)
# where it will then calculate the mean and std err across all the statistics in the model set,
# returning a dictionary that you can throw into randwise_plots where the mean metric is accessable
# via metric name, and the std err is "<metric_name>_err"
# Randwise plots has been intended to be used with this, as all plots are error bars and require an
# error value

precisions = ['32b','12b','6b','4b']  #,'8b']  # What precisions to plot
colors = ['blue', 'green', 'red', 'orange', 'purple', 'pink']  # What colors to use for plots

markers=['D','x','o','v','^','<','>','s','p','P','*','H','h','+',',','X','D','d','|','_','1','2','3','4','8',]
prunevals = ['0.0%','10.0%','20.0%','30.0%','40.0%','50.0%','60.0%','70.0%','80.0%','90.0%','96.6%','98.8%']
marker_lines = []
rand_vals=[0, 50, 75, 90]
kfolds = 4
warnings.filterwarnings("ignore", module="mlines")
for mark, pval in zip(markers, reversed(prunevals)):
    m = mlines.Line2D([], [], color='black', marker=mark, linestyle='None',
                              markersize=10, label=pval)
    marker_lines.append(m)

def gen_mean_err_set(path_str):
    # Path str is like follows: "results_json/all_kfold_json/FT/FT_{}_K{}.json"
    # with the first placeholder being the randomization %, and the second being the fold #
    kf_result_sets = []  # array to hold all of the sets, varying randomness per set
    for rand in rand_vals:
        k_folds = []  # hold all k folds of one model set (rand iteration) of all precisions
        kf_results = {}  # hold mean/err of one model set (rand iteration) of all precisions
        for k in range(1, kfolds + 1):
            with open(path_str.format(rand, k), "r") as read_file:
                k_folds.append(json.load(read_file))
        for p in precisions:
            p_mean = {}  # Holds one precision a model set's results
            for b in range(0, len(k_folds[0][p])):  # Bops count is the key for each entry, but varies
                bops_arr = []
                acc_arr = []
                auc_arr = []
                eff_arr = []
                aiq_arr = []
                bkg_dict = {'j_g': [], 'j_q': [], 'j_w': [], 'j_z': [], 'j_t': []}
                for k in range(0,kfolds):  # I could do this via list comprehension probably, but this works and is easy
                    bops = list(k_folds[k][p])[b]
                    bops_arr.append(int(bops))

                    k_acc = k_folds[k][p][bops]['accuracy']
                    acc_arr.append(k_acc)

                    k_auc = k_folds[k][p][bops]['auc_roc']
                    auc_arr.append(k_auc)

                    k_eff = k_folds[k][p][bops]['net_efficiency']
                    eff_arr.append(k_eff)

                    k_aiq = k_folds[k][p][bops]['AiQ']
                    aiq_arr.append(k_aiq)

                    for particle, value in k_folds[k][p][bops]['sel_bkg_reject'].items():
                        bkg_dict[particle].append(value)

                #print(bops_arr)
                avg_bops = str(np.mean(bops_arr, dtype=int)) # bops can vary slightly across runs due to pruning varying, take avg

                accuracy = np.mean(acc_arr)
                acc_err = np.std(acc_arr) / np.sqrt(kfolds)

                # print(bops, b, len(acc_arr), accuracy, acc_err)

                auc_roc = np.mean(auc_arr)
                auc_roc_err = np.std(auc_arr) / np.sqrt(kfolds)

                aiq = np.mean(aiq_arr)
                aiq_err = np.std(aiq_arr) / np.sqrt(kfolds)

                net_efficiency = np.mean(eff_arr)
                eff_err = np.std(aiq_arr) / np.sqrt(kfolds)

                sel_bkg_reject = {}
                sel_bkg_reject_err = {}

                for particle, value in bkg_dict.items():
                    sel_bkg_reject.update({particle: np.mean(value)})
                    sel_bkg_reject_err.update({particle: np.std(value) / np.sqrt(kfolds)})

                # calculate mean & stderr

                results = {'AiQ': aiq,
                           'accuracy': accuracy,
                           'auc_roc': auc_roc,
                           'net_efficiency': net_efficiency,
                           'sel_bkg_reject': sel_bkg_reject,
                           'AiQ_err': aiq_err,
                           'accuracy_err': acc_err,
                           'auc_roc_err': auc_roc_err,
                           'net_efficiency_err': eff_err,
                           'sel_bkg_reject_err': sel_bkg_reject_err
                           }
                # skip layer metrics
                p_mean.update({avg_bops: results})  # update with each models stats
            kf_results.update({p: p_mean})  # update model set with a precision
        kf_result_sets.append(kf_results)  # update list of all model sets with most recent set
    return kf_result_sets

def randwise_plots(datasets, filename="FT", legendtitle="Fine Tuning Pruning",
                   rand_percent=rand_vals):

    labels_list = ['j_g', 'j_q', 'j_w', 'j_z', 'j_t']
    for FT, rand in zip(datasets, rand_percent):

        for label in labels_list:
            aiq_plot = plt.figure()
            aiq_ax = aiq_plot.add_subplot()
            #aiq_ax.set_title("{}'s Bkgrd Eff. @ Sig Eff = 50% vs BOPS (HLS4ML Jet Tag {}% Rand)".format(label,rand))
            aiq_ax.grid(True)
            aiq_ax.set_xlabel('BOPS')
            aiq_ax.set_ylabel('Background Efficiency @ Signal Efficiency of 50%')
            aiq_ax.set_yscale('log')
            aiq_ax.set_xscale("symlog", linthresh=1e6)
            aiq_ax.set_xlim(1e4, 1e7)
            aiq_ax.set_ylim(10e-4, 10e-1)
            for w, color in zip(precisions, colors):
                aiq_ax.errorbar([int(key) for key in FT[w]],
                                [z['sel_bkg_reject'][label] for z in FT[w].values()],
                                [z['sel_bkg_reject_err'][label] for z in FT[w].values()]
                                , label='{}'.format(w), color=color,capsize=8)
                for x, y, mark in zip([int(key) for key in FT[w]],
                                      [z['sel_bkg_reject'][label] for z in FT[w].values()], markers):
                    aiq_ax.plot(x, y, linestyle='none', marker=mark, markersize=10, color=color, label='_nolegend_')
            aiq_ax.legend(loc='best', title="Class {}".format(label))
            aiq_plot.savefig('bgd_eff_{}_{}_{}rand_BOPS.pdf'.format(label,filename,rand))
            aiq_plot.show()


    #     AUCROC Plot
        eff_plot = plt.figure(figsize=(10,10))
        eff_ax = eff_plot.add_subplot()
        eff_ax.grid(True)
        eff_ax.set_xlabel('BOPs')
        eff_ax.set_ylabel('AUC')
        eff_ax.set_xscale("symlog", linthresh=1e6)
        eff_ax.set_xlim(1e4, 1e7)
        eff_ax.set_ylim(0.45, 1)
        #eff_ax.set_title("AUC vs BOPS {}% Rand (HLS4ML Jet Tagging Model)".format(rand))
        for precision, color in zip(precisions, colors):
            eff_ax.errorbar([int(key) for key in FT[precision]],
                            [z['auc_roc'] for z in FT[precision].values()],
                            [z['auc_roc_err'] for z in FT[precision].values()],
                        label='{}-bit'.format(precision.rstrip('b')),capsize=8,
                        linestyle='solid', color=color)  # , marker='.',markersize=10,
            for x, y, mark in zip([int(key) for key in FT[precision]],
                                  [z['auc_roc'] for z in FT[precision].values()], markers):
                eff_ax.plot(x, y, linestyle='none', marker=mark, markersize=12, color=color, label='_nolegend_')

        eff_ax.add_artist(
            plt.legend(handles=eff_ax.get_lines(), title=legendtitle, loc='lower right', framealpha=0.5))
        #eff_plot.add_artist(plt.legend(handles=marker_lines, title='Percent pruned (approx.)',  bbox_to_anchor=(1.05, 1), loc='upper left', framealpha=0.5))
        eff_plot.savefig('AUCROC_{}_rand{}.pdf'.format(filename,rand))
        # eff_plot.savefig('AUCROC_FT_rand{}.pdf'.format(rand))
        eff_plot.show()

        # Accuracy plot
        eff_plot = plt.figure(figsize=(10,10))
        eff_ax = eff_plot.add_subplot()
        eff_ax.grid(True)
        eff_ax.set_xlabel('BOPs')
        eff_ax.set_ylabel('Accuracy')
        eff_ax.set_xscale("symlog", linthresh=1e6)
        eff_ax.set_xlim(1e4, 1e7)
        eff_ax.set_ylim(0.6, 0.8)
        #eff_ax.set_title("Accuracy vs BOPS {}% Rand (HLS4ML Jet Tagging Model)".format(rand))
        for precision, color in zip(precisions, colors):
            eff_ax.errorbar([int(key) for key in FT[precision]],
                        [z['accuracy'] for z in FT[precision].values()],
                        [z['accuracy_err'] for z in FT[precision].values()],
                        label='{}-bit'.format(precision.rstrip('b')),capsize=8,
                        linestyle='solid', color=color)  # , marker='.',markersize=10,
            for x, y, mark in zip([int(key) for key in FT[precision]],
                                  [z['accuracy'] for z in FT[precision].values()], markers):
                eff_ax.plot(x, y, linestyle='none', marker=mark, markersize=12, color=color, label='_nolegend_')
        eff_ax.add_artist(
            plt.legend(handles=eff_ax.get_lines(), title=legendtitle, loc='lower right', framealpha=0.5))
        #eff_plot.add_artist(plt.legend(handles=marker_lines, title='Percent pruned (approx.)',  bbox_to_anchor=(1.05, 1), loc='upper left',framealpha=0.5))
        eff_plot.savefig('ACC_{}_rand{}.pdf'.format(filename,rand))
        # eff_plot.savefig('ACC_FT_rand{}.pdf'.format(rand))
        eff_plot.show()

        # Efficiency plot
        eff_plot = plt.figure(figsize=(10,10))
        eff_ax = eff_plot.add_subplot()
        eff_ax.grid(True)
        eff_ax.set_xlabel('BOPs')
        eff_ax.set_ylabel('Efficiency')
        eff_ax.set_xscale("symlog", linthresh=1e6)
        eff_ax.set_xlim(1e4, 1e7)
        eff_ax.set_ylim(0, 0.5)
        #eff_ax.set_title("Efficiency vs BOPS {}% Rand (HLS4ML Jet Tagging Model)".format(rand))
        for precision, color in zip(precisions, colors):
            eff_ax.errorbar([int(key) for key in FT[precision]],
                            [z['net_efficiency'] for z in FT[precision].values()],
                            [z['net_efficiency_err'] for z in FT[precision].values()],
                        label='{}-bit'.format(precision.rstrip('b')),capsize=8,
                        linestyle='solid', color=color)  # , marker='.',markersize=10,
            for x, y, mark in zip([int(key) for key in FT[precision]],
                                  [z['net_efficiency'] for z in FT[precision].values()], markers):
                eff_ax.plot(x, y, linestyle='none', marker=mark, markersize=12, color=color, label='_nolegend_')
#            for txt, x, y in zip([str(z['dims']) for z in BO[precision].values()], [int(key) for key in BO[precision]],
#                                 [z['net_efficiency'] for z in BO[precision].values()]):
#                if txt == str(best[precision]):
#                    eff_ax.scatter(x, y, marker="*", s=200, label='Best BO {} ({})'.format(precision, txt), color=color,
#                                  edgecolor='black', zorder=10)
        eff_ax.add_artist(
            plt.legend(handles=eff_ax.get_lines(), title=legendtitle, loc='lower right', framealpha=0.5))
        #eff_plot.add_artist(plt.legend(handles=marker_lines, title='Percent pruned (approx.)',  bbox_to_anchor=(1.05, 1), loc='upper left', framealpha=0.5))
        eff_plot.savefig('Eff_{}_rand{}.pdf'.format(filename,rand))
        # eff_plot.savefig('ACC_FT_rand{}.pdf'.format(rand))
        eff_plot.show()

    for width in precisions:
        eff_plot = plt.figure()
        eff_ax = eff_plot.add_subplot()
        #eff_ax.set_title("AUC ROC vs BOPS ({} HLS4ML Jet Tagging Model)".format(width))
        eff_ax.grid(True)
        eff_ax.set_xlabel('BOPs   ',labelpad=20)
        eff_ax.set_ylabel('AUC ROC')
        if width == "32b":
            eff_ax.set_xscale("symlog", linthresh=1e6)
            eff_ax.set_xlim(1e4, 1e7)
        else:
            eff_ax.set_xscale('linear')
            eff_ax.set_xlim(1e4, 1.1e6)
        eff_ax.set_ylim(0.45, 1)

        for FT, rand, color in zip(datasets, rand_percent, colors):
            print(BO_best[rand][width])
            eff_ax.errorbar([int(key) for key in FT[width]],
                        [z['auc_roc'] for z in FT[width].values()],
                        [z['auc_roc_err'] for z in FT[width].values()],
                        label='{}% Rand'.format(rand),capsize=8,
                        linestyle='solid', color=color)  # , marker='.',markersize=10,
            for x, y, mark in zip([int(key) for key in FT[width]],
                                  [z['auc_roc'] for z in FT[width].values()], markers):
                eff_ax.plot(x, y, linestyle='none', marker=mark, markersize=10, color=color, label='_nolegend_')
            if filename is "FT":

                #txt = [str(z['dims']) for z in BO_best[rand][width].values()]
                x = [int(key) for key in BO_best[rand][width]]
                y = [z['auc_roc'] for z in BO_best[rand][width].values()]
                eff_ax.scatter(x, y, marker="*", s=200, label='Best BO @ {}% Rand'.format(rand), color=color,
                                       edgecolor='black', zorder=10)
        plt.legend(title=legendtitle, loc='best', framealpha=0.5)
        #eff_ax.add_artist(plt.legend(handles=marker_lines, title='Percent pruned (approx.)', framealpha=0.5))
        eff_plot.savefig('AUCROC_randcomp_{}_{}.pdf'.format(filename,width))
        eff_plot.show()

        eff_plot = plt.figure()
        eff_ax = eff_plot.add_subplot()
        #eff_ax.set_title("Efficiency vs BOPS ({} HLS4ML Jet Tagging Model)".format(width))
        eff_ax.grid(True)
        eff_ax.set_xlabel('BOPs   ',labelpad=20)
        eff_ax.set_ylabel('Efficiency')
        if width == "32b":
            eff_ax.set_xscale("symlog", linthresh=1e6)
            eff_ax.set_xlim(1e4, 1e7)
        else:
            eff_ax.set_xscale('linear')
            eff_ax.set_xlim(1e4, 1.1e6)
        eff_ax.set_ylim(0, 0.5)
        for FT, rand, color in zip(datasets, rand_percent, colors):
            eff_ax.errorbar([int(key) for key in FT[width]],
                        [z['net_efficiency'] for z in FT[width].values()],
                        [z['net_efficiency_err'] for z in FT[width].values()],
                        label='{}% Rand'.format(rand), capsize=8,
                        linestyle='solid', color=color)  # , marker='.',markersize=10,
            for x, y, mark in zip([int(key) for key in FT[width]],
                                  [z['net_efficiency'] for z in FT[width].values()], markers):
                eff_ax.plot(x, y, linestyle='none', marker=mark, markersize=10, color=color, label='_nolegend_')
            if filename is "FT":
                #txt = [str(z['dims']) for z in BO_best[rand][width].values()]
                x = [int(key) for key in BO_best[rand][width]]
                y = [z['net_efficiency'] for z in BO_best[rand][width].values()]
                eff_ax.scatter(x, y, marker="*", s=200, label='Best BO @ {}% Rand'.format(rand), color=color,
                               edgecolor='black', zorder=10)
        plt.legend(title=legendtitle, loc='best', framealpha=0.5)
        #eff_ax.add_artist(plt.legend(handles=marker_lines, title='Percent pruned (approx.)', framealpha=0.5))
        eff_plot.savefig('t_eff_randcomp_{}_{}.pdf'.format(filename,width))
        eff_plot.show()

        eff_plot = plt.figure()
        eff_ax = eff_plot.add_subplot()
        #eff_ax.set_title("Accuracy vs BOPS ({} HLS4ML Jet Tagging Model)".format(width))
        eff_ax.grid(True)
        eff_ax.set_xlabel('BOPs   ',labelpad=20)
        eff_ax.set_ylabel('Accuracy')
        if width == "32b":
            eff_ax.set_xscale("symlog", linthresh=1e6)
            eff_ax.set_xlim(1e4, 1e7)
        else:
            eff_ax.set_xscale('linear')
            eff_ax.set_xlim(1e4, 1.1e6)

        if "_TS" in filename:
            eff_ax.set_ylim(0.2, 0.8)
        else:
            eff_ax.set_ylim(0.6, 0.8)

        for FT, rand, color in zip(datasets, rand_percent, colors):
            eff_ax.errorbar([int(key) for key in FT[width]],
                        [z['accuracy'] for z in FT[width].values()],
                        [z['accuracy_err'] for z in FT[width].values()],
                        label='{}% Rand'.format(rand), capsize=8,
                        linestyle='solid', color=color)  # , marker='.',markersize=10,
            for x, y, mark in zip([int(key) for key in FT[width]],
                                  [z['accuracy'] for z in FT[width].values()], markers):
                eff_ax.plot(x, y, linestyle='none', marker=mark, markersize=10, color=color, label='_nolegend_')
            if filename is "FT":
                print(BO_best[rand][width])
                #txt = [str(z['dims']) for z in BO_best[rand][width].values()]
                x = [int(key) for key in BO_best[rand][width]]
                y = [z['accuracy'] for z in BO_best[rand][width].values()]
                eff_ax.scatter(x, y, marker="*", s=200, label='Best BO @ {}% Rand'.format(rand), color=color,
                               edgecolor='black', zorder=10)
        plt.legend(title=legendtitle, loc='best', framealpha=0.5)
        #eff_ax.add_artist(plt.legend(handles=marker_lines, title='Percent pruned (approx.)', framealpha=0.5))
        eff_plot.savefig('acc_randcomp_{}_{}.pdf'.format(filename, width))
        eff_plot.show()

if __name__ == "__main__":
    # with open("../results_json/rand_trials_250e/FT_0.json", "r") as read_file:
    #     FT_0 = json.load(read_file)
    #
    # with open("../results_json/rand_trials_250e/FT_25.json", "r") as read_file:
    #     FT_25 = json.load(read_file)
    #
    # with open("../results_json/rand_trials_250e/FT_50.json", "r") as read_file:
    #     FT_50 = json.load(read_file)
    #
    # with open("../results_json/rand_trials_250e/FT_75.json", "r") as read_file:
    #     FT_75 = json.load(read_file)
    #
    # with open("../results_json/rand_trials_250e/FT_90.json", "r") as read_file:
    #     FT_90 = json.load(read_file)
    #
    # with open("../results_json/rand_trials_250e/FT_100.json", "r") as read_file:
    #     FT_100 = json.load(read_file)
    #
    #
    #
    # with open("../results_json/BN_trials_250e/FT_0_BNNoStat.json", "r") as read_file:
    #     FT_0_BNNoStat = json.load(read_file)
    #
    #
    #
    # with open("../results_json/BN_trials_250e/FT_0_NoBN.json", "r") as read_file:
    #     FT_0_NoBN = json.load(read_file)
    #
    # with open("../results_json/BN_trials_250e/FT_25_NoBN.json", "r") as read_file:
    #     FT_25_NoBN = json.load(read_file)
    #
    # with open("../results_json/BN_trials_250e/FT_50_NoBN.json", "r") as read_file:
    #     FT_50_NoBN = json.load(read_file)
    #
    # with open("../results_json/BN_trials_250e/FT_75_NoBN.json", "r") as read_file:
    #     FT_75_NoBN = json.load(read_file)
    #
    # with open("../results_json/BN_trials_250e/FT_90_NoBN.json", "r") as read_file:
    #     FT_90_NoBN = json.load(read_file)

    #with open("../results_json/BN_trials_250e/FT_100_NoBN.json", "r") as read_file:
   #     FT_100_NoBN = json.load(read_file)


   # with open("../results_json/BN_trials_250e/FT_0_NoL1.json", "r") as read_file:
   #     FT_0_NoL1 = json.load(read_file)

   # with open("../results_json/BN_trials_250e/FT_25_NoL1.json", "r") as read_file:
   #     FT_25_NoL1 = json.load(read_file)

   # with open("../results_json/BN_trials_250e/FT_50_NoL1.json", "r") as read_file:
   #     FT_50_NoL1 = json.load(read_file)

   # with open("../results_json/BN_trials_250e/FT_75_NoL1.json", "r") as read_file:
   #     FT_75_NoL1 = json.load(read_file)

   #with open("../results_json/BN_trials_250e/FT_90_NoL1.json", "r") as read_file:
   #    FT_90_NoL1 = json.load(read_file)

   # with open("../results_json/BN_trials_250e/FT_100_NoL1.json", "r") as read_file:
   #     FT_100_NoL1 = json.load(read_file)


   # with open("../results_json/BN_trials_250e/LT_0_BNNoStat.json", "r") as read_file:
   #     LT_0_BNNoStat = json.load(read_file)

    with open("../results_json/BO_Redo/BO_Redo.json", "r") as read_file:
        BO = json.load(read_file)

    with open("../results_json/BO_Redo/BO_6b_Wide.json", "r") as read_file:
        BO_6b_wide = json.load(read_file)

    with open("../results_json/BO_Redo/BO_combo.json", "r") as read_file:
        BO_combo = json.load(read_file)

    BO_best = {r:{} for r in rand_vals}
    for rand in BO_best:
        with open("../results_json/BO_Redo/BO_Best_{}_AiQ.json".format(rand), "r") as read_file:
            BO_best[rand].update(json.load(read_file))



   # FT_Trainset = []
   # for rand in rand_vals:
   #     with open("../results_json/rand_trials_250e/TrainSet/FT_{}_TRAINSET.json".format(rand), "r") as read_file:
   #        FT_Trainset.append(json.load(read_file))
   # FT_NoBN_Trainset = []
   # for rand in rand_vals:
   #     with open("../results_json/rand_trials_250e/TrainSet/FT_{}_NoBN_TRAINSET.json".format(rand), "r") as read_file:
   #         FT_NoBN_Trainset.append(json.load(read_file))
   # FT_NoL1_Trainset = []
   # for rand in rand_vals:
   #     with open("../results_json/rand_trials_250e/TrainSet/FT_{}_NoL1_TRAINSET.json".format(rand), "r") as read_file:
   #         FT_NoL1_Trainset.append(json.load(read_file))

    FT_kf = gen_mean_err_set("../results_json/all_kfold_json/FT/FT_{}_K{}.json")
    FT_NoBN_kf = gen_mean_err_set("../results_json/all_kfold_json/FT_NoBN/FT_{}_NoBN_K{}.json")
    FT_NoL1_kf = gen_mean_err_set("../results_json/all_kfold_json/FT_NoL1/FT_{}_NoL1_K{}.json")
    LT_kf = gen_mean_err_set("../results_json/all_kfold_json/LT/LT_{}_K{}.json")

    colors = ['blue', 'green', 'red', 'orange', 'purple']

    #Normal, "Test Set" runs - Comment out lines to disable generation of that plot set

    #randwise_plots(FT_kf, "FT")
    #randwise_plots(FT_NoBN_kf, "FT_NoBN", "FT, No BatNorm")
    #randwise_plots(FT_NoL1_kf, "FT_NoL1", "FT, No L1 Reg")
    #randwise_plots(LT_kf, "LT")

    best = {'4b': [44, 32, 32],
            '6b': [54, 32, 32],
            '12b': [64, 32, 19],
            '32b': [64, 28, 27]}

    # BO 4b vs 32b
    # Accuracy plot
    eff_plot = plt.figure()
    eff_ax = eff_plot.add_subplot()
    eff_ax.grid(True)
    eff_ax.set_xlabel('BOPs')
    eff_ax.set_ylabel('Accuracy')
    eff_ax.set_xscale("symlog", linthresh=1e6)
    eff_ax.set_xlim(1e4, 1e7)
    eff_ax.set_ylim(0.6, 0.8)
    # eff_ax.set_title("Accuracy vs BOPS (HLS4ML Jet Tagging Model)")
    eff_ax.errorbar([int(key) for key in FT_kf[0]['32b']],
                    [z['accuracy'] for z in FT_kf[0]['32b'].values()],
                    [z['accuracy_err'] for z in FT_kf[0]['32b'].values()],
                    label='32b FT', capsize=8,
                    linestyle='solid', color='red')  # , marker='.',markersize=10,
    for x, y, mark in zip([int(key) for key in FT_kf[0]['32b']],
                          [z['accuracy'] for z in FT_kf[0]['32b'].values()], markers):
        eff_ax.plot(x, y, linestyle='solid', marker=mark, markersize=10, color='red', label='_nolegend_')

    eff_ax.errorbar([int(key) for key in FT_kf[0]['4b']],
                    [z['accuracy'] for z in FT_kf[0]['4b'].values()],
                    [z['accuracy_err'] for z in FT_kf[0]['4b'].values()],
                    label='4b FT', capsize=8,
                    linestyle='solid', color='blue')  # , marker='.',markersize=10,
    for x, y, mark in zip([int(key) for key in FT_kf[0]['4b']],
                          [z['accuracy'] for z in FT_kf[0]['4b'].values()], markers):
        eff_ax.plot(x, y, linestyle='none', marker=mark, markersize=10, color='blue', label='_nolegend_')

    eff_ax.scatter([int(key) for key in BO['32b']], [z['accuracy'] for z in BO['32b'].values()],
                   label='32b BO', color='red', s=200,
                   alpha=0.1)  # , marker='.',markersize=10,
    for txt, x, y, isbest in zip([str(z['dims']) for z in BO['32b'].values()], [int(key) for key in BO['32b']],
                                 [z['accuracy'] for z in BO['32b'].values()],
                                 [z['best'] for z in BO['32b'].values()]):
        if isbest:
            eff_ax.annotate(txt, (x, y), color='red', label='_nolegend_', size=24)
            eff_ax.scatter(x, y, marker="*", s=250, label='Best BO {} ({})'.format('32b', txt), color='red',
                           edgecolor='blue', zorder=10)

    eff_ax.scatter([int(key) for key in BO['4b']], [z['accuracy'] for z in BO['4b'].values()],
                   label='4b BO', color='blue', s=200,
                   alpha=0.1)  # , marker='.',markersize=10,
    for txt, x, y, isbest in zip([str(z['dims']) for z in BO['4b'].values()], [int(key) for key in BO['4b']],
                                 [z['accuracy'] for z in BO['4b'].values()],
                                 [z['best'] for z in BO['4b'].values()]):
        if isbest:
            #eff_ax.annotate(txt, (x, y), color='blue', label='_nolegend_', size=24)
            eff_ax.scatter(x, y, marker="*", s=250, label='Best BO {} ({})'.format('4b', txt), color='blue',
                           edgecolor='red', zorder=10)
    eff_ax.legend(title='Optimization Technique', loc='best', framealpha=0.5)
    # eff_ax.add_artist(
    #    plt.legend(handles=marker_lines, title='Percent pruned (approx.)', loc='upper right', framealpha=0.5))
    eff_plot.savefig('ACC_BOFT_32_4.pdf')
    # eff_plot.savefig('ACC_FT_BNcomp.pdf')
    eff_plot.show()


    # BO 6b vs 32b
    # Accuracy plot
    eff_plot = plt.figure()
    eff_ax = eff_plot.add_subplot()
    eff_ax.grid(True)
    eff_ax.set_xlabel('BOPs')
    eff_ax.set_ylabel('Accuracy')
    eff_ax.set_xscale("symlog", linthresh=1e6)
    eff_ax.set_xlim(1e4, 1e7)
    eff_ax.set_ylim(0.6, 0.8)
    # eff_ax.set_title("Accuracy vs BOPS (HLS4ML Jet Tagging Model)")
    eff_ax.errorbar([int(key) for key in FT_kf[0]['32b']],
                    [z['accuracy'] for z in FT_kf[0]['32b'].values()],
                    [z['accuracy_err'] for z in FT_kf[0]['32b'].values()],
                    label='32b FT', capsize=8,
                    linestyle='solid', color='red')  # , marker='.',markersize=10,
    for x, y, mark in zip([int(key) for key in FT_kf[0]['32b']],
                          [z['accuracy'] for z in FT_kf[0]['32b'].values()], markers):
        eff_ax.plot(x, y, linestyle='solid', marker=mark, markersize=10, color='red', label='_nolegend_')

    eff_ax.errorbar([int(key) for key in FT_kf[0]['6b']],
                    [z['accuracy'] for z in FT_kf[0]['6b'].values()],
                    [z['accuracy_err'] for z in FT_kf[0]['6b'].values()],
                    label='6b FT', capsize=8,
                    linestyle='solid', color='blue')  # , marker='.',markersize=10,
    for x, y, mark in zip([int(key) for key in FT_kf[0]['6b']],
                          [z['accuracy'] for z in FT_kf[0]['6b'].values()], markers):
        eff_ax.plot(x, y, linestyle='none', marker=mark, markersize=10, color='blue', label='_nolegend_')

    eff_ax.scatter([int(key) for key in BO['32b']], [z['accuracy'] for z in BO['32b'].values()],
                   label='32b BO', color='red', s=200,
                   alpha=0.1)  # , marker='.',markersize=10,
    for txt, x, y, isbest in zip([str(z['dims']) for z in BO['32b'].values()], [int(key) for key in BO['32b']],
                                 [z['accuracy'] for z in BO['32b'].values()],
                                 [z['best'] for z in BO['32b'].values()]):
        if isbest:
            #eff_ax.annotate(txt, (x, y), color='red', label='_nolegend_', size=24)
            eff_ax.scatter(x, y, marker="*", s=250, label='Best BO {} ({})'.format('32b', txt), color='red',
                           edgecolor='blue', zorder=10)

    eff_ax.scatter([int(key) for key in BO['6b']], [z['accuracy'] for z in BO['6b'].values()],
                   label='6b BO', color='blue', s=200,
                   alpha=0.1)  # , marker='.',markersize=10,
    for txt, x, y, isbest in zip([str(z['dims']) for z in BO['6b'].values()], [int(key) for key in BO['6b']],
                                 [z['accuracy'] for z in BO['6b'].values()],
                                 [z['best'] for z in BO['6b'].values()]):
        if isbest:
            #eff_ax.annotate(txt, (x, y), color='blue', label='_nolegend_', size=24)
            eff_ax.scatter(x, y, marker="*", s=250, label='Best BO {} ({})'.format('6b', txt), color='blue',
                           edgecolor='red', zorder=10)

    eff_ax.scatter([int(key) for key in BO_6b_wide['6b']], [z['accuracy'] for z in BO_6b_wide['6b'].values()],
                   label='6b BO (Wide Scan)', color='green', s=200,
                   alpha=0.1)  # , marker='.',markersize=10,
    for txt, x, y, isbest in zip([str(z['dims']) for z in BO_6b_wide['6b'].values()], [int(key) for key in BO_6b_wide['6b']],
                                 [z['accuracy'] for z in BO_6b_wide['6b'].values()],
                                 [z['best'] for z in BO_6b_wide['6b'].values()]):
        if isbest:
            # eff_ax.annotate(txt, (x, y), color='blue', label='_nolegend_', size=24)
            eff_ax.scatter(x, y, marker="*", s=250, label='Best BO {} ({})'.format('6b', txt), color='green',
                           edgecolor='red', zorder=10)
    eff_ax.legend(title='Optimization Technique', loc='best', framealpha=0.5)
    # eff_ax.add_artist(
    #    plt.legend(handles=marker_lines, title='Percent pruned (approx.)', loc='upper right', framealpha=0.5))
    eff_plot.savefig('ACC_BOFT_32_6.pdf')
    # eff_plot.savefig('ACC_FT_BNcomp.pdf')
    eff_plot.show()

    # BO 6b vs 32b
    # Accuracy plot
    eff_plot = plt.figure()
    eff_ax = eff_plot.add_subplot()
    eff_ax.grid(True)
    eff_ax.set_xlabel('BOPs')
    eff_ax.set_ylabel('Accuracy')
    eff_ax.set_xscale("symlog", linthresh=1e6)
    eff_ax.set_xlim(1e4, 1e7)
    eff_ax.set_ylim(0.6, 0.8)
    # eff_ax.set_title("Accuracy vs BOPS (HLS4ML Jet Tagging Model)")
    eff_ax.errorbar([int(key) for key in FT_kf[0]['32b']],
                    [z['accuracy'] for z in FT_kf[0]['32b'].values()],
                    [z['accuracy_err'] for z in FT_kf[0]['32b'].values()],
                    label='32b FT', capsize=8,
                    linestyle='solid', color='red')  # , marker='.',markersize=10,
    for x, y, mark in zip([int(key) for key in FT_kf[0]['32b']],
                          [z['accuracy'] for z in FT_kf[0]['32b'].values()], markers):
        eff_ax.plot(x, y, linestyle='solid', marker=mark, markersize=10, color='red', label='_nolegend_')

    eff_ax.errorbar([int(key) for key in FT_kf[0]['6b']],
                    [z['accuracy'] for z in FT_kf[0]['6b'].values()],
                    [z['accuracy_err'] for z in FT_kf[0]['6b'].values()],
                    label='6b FT', capsize=8,
                    linestyle='solid', color='blue')  # , marker='.',markersize=10,
    for x, y, mark in zip([int(key) for key in FT_kf[0]['6b']],
                          [z['accuracy'] for z in FT_kf[0]['6b'].values()], markers):
        eff_ax.plot(x, y, linestyle='none', marker=mark, markersize=10, color='blue', label='_nolegend_')

    eff_ax.scatter([int(key) for key in BO['32b']], [z['accuracy'] for z in BO['32b'].values()],
                   label='32b BO', color='red', s=200,
                   alpha=0.1)  # , marker='.',markersize=10,
    for txt, x, y, isbest in zip([str(z['dims']) for z in BO['32b'].values()], [int(key) for key in BO['32b']],
                                 [z['accuracy'] for z in BO['32b'].values()],
                                 [z['best'] for z in BO['32b'].values()]):
        if isbest:
            # eff_ax.annotate(txt, (x, y), color='red', label='_nolegend_', size=24)
            eff_ax.scatter(x, y, marker="*", s=250, label='Best BO {} ({})'.format('32b', txt), color='red',
                           edgecolor='blue', zorder=10)

    eff_ax.scatter([int(key) for key in BO_combo['6b']], [z['accuracy'] for z in BO_combo['6b'].values()],
                   label='6b BO (combo)', color='blue', s=200,
                   alpha=0.1)  # , marker='.',markersize=10,
    for txt, x, y, isbest in zip([str(z['dims']) for z in BO_combo['6b'].values()], [int(key) for key in BO_combo['6b']],
                                 [z['accuracy'] for z in BO_combo['6b'].values()],
                                 [z['best'] for z in BO_combo['6b'].values()]):
        if isbest:
            # eff_ax.annotate(txt, (x, y), color='blue', label='_nolegend_', size=24)
            eff_ax.scatter(x, y, marker="*", s=250, label='Best BO {} ({})'.format('6b', txt), color='blue',
                           edgecolor='red', zorder=10)
    eff_ax.legend(title='Optimization Technique', loc='best', framealpha=0.5)
    # eff_ax.add_artist(
    #    plt.legend(handles=marker_lines, title='Percent pruned (approx.)', loc='upper right', framealpha=0.5))
    eff_plot.savefig('ACC_BOFT_32_6_combo.pdf')
    # eff_plot.savefig('ACC_FT_BNcomp.pdf')
    eff_plot.show()


    # BO 6b vs 32b
    # Accuracy plot
    eff_plot = plt.figure()
    eff_ax = eff_plot.add_subplot()
    eff_ax.grid(True)
    eff_ax.set_xlabel('BOPs')
    eff_ax.set_ylabel('Accuracy')
    eff_ax.set_xscale("symlog", linthresh=1e6)
    eff_ax.set_xlim(1e4, 1e7)
    eff_ax.set_ylim(0.6, 0.8)
    # eff_ax.set_title("Accuracy vs BOPS (HLS4ML Jet Tagging Model)")
    eff_ax.errorbar([int(key) for key in FT_kf[0]['32b']],
                    [z['accuracy'] for z in FT_kf[0]['32b'].values()],
                    [z['accuracy_err'] for z in FT_kf[0]['32b'].values()],
                    label='32b FT', capsize=8,
                    linestyle='solid', color='red')  # , marker='.',markersize=10,
    for x, y, mark in zip([int(key) for key in FT_kf[0]['32b']],
                          [z['accuracy'] for z in FT_kf[0]['32b'].values()], markers):
        eff_ax.plot(x, y, linestyle='solid', marker=mark, markersize=10, color='red', label='_nolegend_')

    eff_ax.errorbar([int(key) for key in FT_kf[0]['12b']],
                    [z['accuracy'] for z in FT_kf[0]['12b'].values()],
                    [z['accuracy_err'] for z in FT_kf[0]['12b'].values()],
                    label='12b FT', capsize=8,
                    linestyle='solid', color='blue')  # , marker='.',markersize=10,
    for x, y, mark in zip([int(key) for key in FT_kf[0]['12b']],
                          [z['accuracy'] for z in FT_kf[0]['12b'].values()], markers):
        eff_ax.plot(x, y, linestyle='none', marker=mark, markersize=10, color='blue', label='_nolegend_')

    eff_ax.scatter([int(key) for key in BO['32b']], [z['accuracy'] for z in BO['32b'].values()],
                   label='32b BO', color='red', s=200,
                   alpha=0.1)  # , marker='.',markersize=10,
    for txt, x, y, isbest in zip([str(z['dims']) for z in BO['32b'].values()], [int(key) for key in BO['32b']],
                                 [z['accuracy'] for z in BO['32b'].values()],
                                 [z['best'] for z in BO['32b'].values()]):
        if isbest:
            eff_ax.annotate(txt, (x, y), color='red', label='_nolegend_', size=24)
            eff_ax.scatter(x, y, marker="*", s=250, label='Best BO {} ({})'.format('32b', txt), color='red',
                           edgecolor='blue', zorder=10)

    eff_ax.scatter([int(key) for key in BO['12b']], [z['accuracy'] for z in BO['12b'].values()],
                   label='12b BO', color='blue', s=200,
                   alpha=0.1)  # , marker='.',markersize=10,
    for txt, x, y, isbest in zip([str(z['dims']) for z in BO['12b'].values()], [int(key) for key in BO['12b']],
                                 [z['accuracy'] for z in BO['12b'].values()],
                                 [z['best'] for z in BO['12b'].values()]):
        if isbest:
            #eff_ax.annotate(txt, (x, y), color='blue', label='_nolegend_', size=24)
            eff_ax.scatter(x, y, marker="*", s=250, label='Best BO {} ({})'.format('12b', txt), color='blue',
                           edgecolor='red', zorder=10)
    eff_ax.legend(title='Optimization Technique', loc='best', framealpha=0.5)
    # eff_ax.add_artist(
    #    plt.legend(handles=marker_lines, title='Percent pruned (approx.)', loc='upper right', framealpha=0.5))
    eff_plot.savefig('ACC_BOFT_32_12.pdf')
    # eff_plot.savefig('ACC_FT_BNcomp.pdf')
    eff_plot.show()

    labels_list = ['j_g', 'j_q', 'j_w', 'j_z', 'j_t']
    for label in labels_list:
        print(label)
        aiq_plot = plt.figure()
        aiq_ax = aiq_plot.add_subplot()
        # aiq_ax.set_title("{}'s Bkgrd Eff. @ Sig Eff = 50% vs BOPS (HLS4ML Jet Tag {}% Rand)".format(label,rand))
        aiq_ax.grid(True)
        aiq_ax.set_xlabel('BOPS')
        aiq_ax.set_ylabel('Background Efficiency @ Signal Efficiency of 50%')
        aiq_ax.set_yscale('log')
        aiq_ax.set_xscale("symlog", linthresh=1e6)
        aiq_ax.set_xlim(1e4, 1e7)
        aiq_ax.set_ylim(10e-4, 10e-1)
        bo_colors = ['red', 'blue']
        bo_edgecolors =  reversed(bo_colors)
        bo_precisions = ['32b', '6b']
        for w, color, ec in zip(bo_precisions, bo_colors, bo_edgecolors):
            # FT
            aiq_ax.errorbar([int(key) for key in FT_kf[0][w]],
                            [z['sel_bkg_reject'][label] for z in FT_kf[0][w].values()],
                            [z['sel_bkg_reject_err'][label] for z in FT_kf[0][w].values()]
                            , label='{}'.format(w), color=color, capsize=8)
            for x, y, mark in zip([int(key) for key in FT_kf[0][w]],
                                  [z['sel_bkg_reject'][label] for z in FT_kf[0][w].values()], markers):
                aiq_ax.plot(x, y, linestyle='none', marker=mark, markersize=10, color=color, label='_nolegend_')
            # BO
            aiq_ax.scatter([int(key) for key in BO[w]], [z['sel_bkg_reject'][label] for z in BO[w].values()],
                           label='{} BO'.format(w), color=color, s=200,
                           alpha=0.1)  # , marker='.',markersize=10,
            for txt, x, y, isbest in zip([str(z['dims']) for z in BO[w].values()], [int(key) for key in BO[w]],
                                 [z['sel_bkg_reject'][label] for z in BO[w].values()],
                                 [z['best'] for z in BO[w].values()]):
                if isbest:
                    aiq_ax.scatter(x, y, marker="*", s=250, label='Best BO {} ({})'.format('6b', txt), color=color,
                                   edgecolor=ec, zorder=10)
        aiq_ax.legend(loc='upper right', title="Class {}".format(label))
        aiq_plot.savefig('bgd_eff_{}_32_6_BOFT_BOPS.pdf'.format(label))
        aiq_plot.show()

    # ~~~~~~~~~~~~~~~~~ LT vs FT Plots ~~~~~~~~~~~~~~~~~~~~

    fig = pylab.figure(figsize=(13,1.75))
    leg_artist = pylab.figlegend(handles=marker_lines, title='Percent pruned (approx.)', framealpha=0.5, ncol=math.ceil(len(marker_lines)/2.0))
    leg_artist._legend_box.align = 'center'
    artist_handle = fig.add_artist(leg_artist)  # ,  bbox_to_anchor=(1.05, 1), loc='upper left'
    fig.savefig("PruningLegend.pdf")
    fig.show()

    labels_list = ['j_g', 'j_q', 'j_w', 'j_z', 'j_t']
    for label in labels_list:
        aiq_plot = plt.figure()
        aiq_ax = aiq_plot.add_subplot()
        # aiq_ax.set_title("{}'s Bkgrd Eff. @ Sig Eff = 50% vs BOPS (HLS4ML Jet Tag {}% Rand)".format(label,rand))
        aiq_ax.grid(True)
        aiq_ax.set_xlabel('BOPS')
        aiq_ax.set_ylabel('Background Efficiency @ Signal Efficiency of 50%')
        aiq_ax.set_yscale('log')
        aiq_ax.set_xscale("symlog", linthresh=1e6)
        aiq_ax.set_xlim(1e4, 1e7)
        aiq_ax.set_ylim(10e-4, 10e-1)
        for w, color in zip(precisions, colors):
            aiq_ax.errorbar([int(key) for key in FT_kf[0][w]],
                            [z['sel_bkg_reject'][label] for z in FT_kf[0][w].values()],
                            [z['sel_bkg_reject_err'][label] for z in FT_kf[0][w].values()]
                            , label='{}-bit FT'.format(w.rstrip('b')), color=color, capsize=8, linestyle='solid')
            for x, y, mark in zip([int(key) for key in FT_kf[0][w]],
                                  [z['sel_bkg_reject'][label] for z in FT_kf[0][w].values()], markers):
                aiq_ax.plot(x, y, linestyle='none', marker=mark, markersize=10, color=color, label='_nolegend_')
            aiq_ax.errorbar([int(key) for key in LT_kf[0][w]],
                            [z['sel_bkg_reject'][label] for z in LT_kf[0][w].values()],
                            [z['sel_bkg_reject_err'][label] for z in LT_kf[0][w].values()]
                            , label='{}-bit LT'.format(w.rstrip('b')), color=color, capsize=8, linestyle='dotted')
            for x, y, mark in zip([int(key) for key in LT_kf[0][w]],
                                  [z['sel_bkg_reject'][label] for z in LT_kf[0][w].values()], markers):
                aiq_ax.plot(x, y, linestyle='none', marker=mark, markersize=10, color=color, label='_nolegend_')
        aiq_ax.legend(loc='best', title="Class {}".format(label))
        aiq_plot.savefig('bgd_eff_{}_FT_vs_LT_0rand_BOPS.pdf'.format(label))
        aiq_plot.show()

    # Accuracy plot
    eff_plot = plt.figure(figsize=(10,10))
    eff_ax = eff_plot.add_subplot()
    eff_ax.grid(True)
    eff_ax.set_xlabel('BOPs')
    eff_ax.set_ylabel('Accuracy')
    eff_ax.set_xscale("symlog", linthresh=1e6)
    eff_ax.set_xlim(1e4, 1e7)
    eff_ax.set_ylim(0.6, 0.8)
    #eff_ax.set_title("Accuracy vs BOPS {}% Rand (HLS4ML Jet Tagging Model)".format(0))
    lines=[]
    for precision, color in zip(precisions, colors):
        lines.append(eff_ax.errorbar([int(key) for key in FT_kf[0][precision]],
                                 [z['accuracy'] for z in FT_kf[0][precision].values()],
                                 [z['accuracy_err'] for z in FT_kf[0][precision].values()],
                                 label='{}-bit FT'.format(precision.rstrip('b')), capsize=8,
                                 linestyle='solid', color=color)[0])
        for x, y, mark in zip([int(key) for key in FT_kf[0][precision]],
                              [z['accuracy'] for z in FT_kf[0][precision].values()], markers):
            eff_ax.plot(x, y, linestyle='none', marker=mark, markersize=10, color=color, label='_nolegend_')

        lines.append(eff_ax.errorbar([int(key) for key in LT_kf[0][precision]],
                        [z['accuracy'] for z in LT_kf[0][precision].values()],
                        [z['accuracy_err'] for z in LT_kf[0][precision].values()],
                    label='{}-bit LT'.format(precision.rstrip('b')), capsize=8,
                    linestyle='dotted', color=color)[0])  # , marker='.',markersize=10,
        for x, y, mark in zip([int(key) for key in LT_kf[0][precision]],
                              [z['accuracy'] for z in LT_kf[0][precision].values()], markers):
            eff_ax.plot(x, y, linestyle='none', marker=mark, markersize=10, color=color, label='_nolegend_')
    eff_ax.legend(title='Pruning Style', loc='lower right', framealpha=0.5)
    #eff_plot.add_artist(plt.legend(handles=marker_lines, title='Percent pruned (approx.)',  bbox_to_anchor=(1.05, 1), loc='upper left', framealpha=0.5))
    eff_plot.savefig('ACC_FT_vs_LT_0Rand.pdf', bbox_inches='tight')
    # eff_plot.savefig('ACC_FT_rand{}.pdf'.format(rand))
    eff_plot.show()

    # Efficiency plot
    eff_plot = plt.figure(figsize=(10,10))
    eff_ax = eff_plot.add_subplot()
    eff_ax.grid(True)
    eff_ax.set_xlabel('BOPs')
    eff_ax.set_ylabel('Efficiency')
    eff_ax.set_xscale("symlog", linthresh=1e6)
    eff_ax.set_xlim(1e4, 1e7)
    eff_ax.set_ylim(0, 0.5)
    #eff_ax.set_title("Efficiency vs BOPS {}% Rand (HLS4ML Jet Tagging Model)".format(0))
    for precision, color in zip(precisions, colors):
        eff_ax.errorbar([int(key) for key in FT_kf[0][precision]],
                    [z['net_efficiency'] for z in FT_kf[0][precision].values()],
                    [z['net_efficiency_err'] for z in FT_kf[0][precision].values()],
                    label='{}-bit FT'.format(precision.rstrip('b')), capsize=8,
                    linestyle='solid', color=color)  # , marker='.',markersize=10,
        for x, y, mark in zip([int(key) for key in FT_kf[0][precision]],
                              [z['net_efficiency'] for z in FT_kf[0][precision].values()], markers):
            eff_ax.plot(x, y, linestyle='none', marker=mark, markersize=10, color=color, label='_nolegend_')
        eff_ax.errorbar([int(key) for key in LT_kf[0][precision]],
                        [z['net_efficiency'] for z in LT_kf[0][precision].values()],
                        [z['net_efficiency_err'] for z in LT_kf[0][precision].values()],
                    label='{}-bit LT'.format(precision.rstrip('b')), capsize=8,
                    linestyle='dotted', color=color)  # , marker='.',markersize=10,
        for x, y, mark in zip([int(key) for key in LT_kf[0][precision]],
                              [z['net_efficiency'] for z in LT_kf[0][precision].values()], markers):
            eff_ax.plot(x, y, linestyle='none', marker=mark, markersize=10, color=color, label='_nolegend_')
    eff_ax.legend(title='Pruning Style', loc='lower right', framealpha=0.5)
    #eff_plot.add_artist(plt.legend(handles=marker_lines, title='Percent pruned (approx.)',  bbox_to_anchor=(1.05, 1), loc='upper left', framealpha=0.5))
    eff_plot.savefig('Eff_FT_vs_LT_0Rand.pdf')
    # eff_plot.savefig('ACC_FT_rand{}.pdf'.format(rand))
    eff_plot.show()




    #print(BO_best)
    #best = {'4b': [44, 32, 32],
    #        '6b': [54, 32, 32],
    #        '12b': [64, 32, 19],
    #        '32b': [64, 28, 27]}

    #ft_models = [FT_0, FT_50, FT_75, FT_90]
    #NoBN_Models = [FT_0_NoBN,  FT_50_NoBN, FT_75_NoBN, FT_90_NoBN]
    #NoL1_Models = [FT_0_NoL1, FT_50_NoL1,FT_75_NoL1, FT_90_NoL1]
    #colors = ['blue', 'green', 'red', 'orange', 'purple']
    #Normal, "Test Set" runs
    #randwise_plots(ft_models, "FT")
    #randwise_plots(NoBN_Models, "FT_NoBN", "FT, No BatNorm")
    #randwise_plots(NoL1_Models, "FT_NoL1", "FT, No L1 Reg")

    #AIQ calc done on Training set
    #randwise_plots(FT_Trainset, "FT_TS", "FT (TS)")
    #randwise_plots(FT_NoBN_Trainset, "FT_NoBN_TS", "FT, No BatNorm (TS)")
    #randwise_plots(FT_NoL1_Trainset, "FT_NoL1_TS", "FT, No L1 Reg (TS)")

    # ~~~~~~~~~~~~~~~~~~~ BN & L1 Comparsion plots ~~~~~~~~~~~~~~~~~~~~

    # AUCROC Plot
    eff_plot = plt.figure()
    eff_ax = eff_plot.add_subplot()
    eff_ax.grid(True)
    eff_ax.set_xlabel('BOPs')
    eff_ax.set_ylabel('AUC')
    eff_ax.set_xscale("symlog", linthresh=1e6)
    eff_ax.set_xlim(1e4, 1e7)
    eff_ax.set_ylim(0.45, 1)
    #eff_ax.set_title("AUC vs BOPS (HLS4ML Jet Tagging Model)")
    eff_ax.errorbar([int(key) for key in FT_kf[0]['32b']],
                [z['auc_roc'] for z in FT_kf[0]['32b'].values()],
                [z['auc_roc_err'] for z in FT_kf[0]['32b'].values()],
                label='32b FT w/ Batch Norm', capsize=8,
                linestyle='solid', color='red')  # , marker='.',markersize=10,
    for x, y, mark in zip([int(key) for key in FT_kf[0]['32b']],
                          [z['auc_roc'] for z in FT_kf[0]['32b'].values()], markers):
        eff_ax.plot(x, y, linestyle='solid', marker=mark, markersize=10, color='red', label='_nolegend_')

    eff_ax.errorbar([int(key) for key in FT_NoBN_kf[0]['32b']],
                [z['auc_roc'] for z in FT_NoBN_kf[0]['32b'].values()],
                [z['auc_roc_err'] for z in FT_NoBN_kf[0]['32b'].values()],
                label='32b FT No Batch Norm', capsize=8,
                linestyle='solid', color='blue')  # , marker='.',markersize=10,
    for x, y, mark in zip([int(key) for key in FT_NoBN_kf[0]['32b']],
                          [z['auc_roc'] for z in FT_NoBN_kf[0]['32b'].values()], markers):
        eff_ax.plot(x, y, linestyle='none', marker=mark, markersize=10, color='blue', label='_nolegend_')

    eff_ax.errorbar([int(key) for key in FT_kf[0]['6b']],
                [z['auc_roc'] for z in FT_kf[0]['6b'].values()],
                [z['auc_roc_err'] for z in FT_kf[0]['6b'].values()],
                label='6b FT w/ Batch Norm', capsize=8,
                linestyle='dotted', color='red')  # , marker='.',markersize=10,
    for x, y, mark in zip([int(key) for key in FT_kf[0]['6b']],
                          [z['auc_roc'] for z in FT_kf[0]['6b'].values()], markers):
        eff_ax.plot(x, y, linestyle='none', marker=mark, markersize=10, color='red', label='_nolegend_')

    eff_ax.errorbar([int(key) for key in FT_NoBN_kf[0]['6b']],
                [z['auc_roc'] for z in FT_NoBN_kf[0]['6b'].values()],
                [z['auc_roc_err'] for z in FT_NoBN_kf[0]['6b'].values()],
                label='6b FT No Batch Norm', capsize=8,
                linestyle='dotted', color='blue')  # , marker='.',markersize=10,
    for x, y, mark in zip([int(key) for key in FT_NoBN_kf[0]['6b']],
                          [z['auc_roc'] for z in FT_NoBN_kf[0]['6b'].values()], markers):
        eff_ax.plot(x, y, linestyle='none', marker=mark, markersize=10, color='blue', label='_nolegend_')
    eff_ax.legend(title='Fine tuning pruning', loc='lower left', framealpha=0.5)
    #eff_ax.add_artist(
    #    plt.legend(handles=marker_lines, title='Percent pruned (approx.)', loc='upper right', framealpha=0.5))
    eff_plot.savefig('AUCROC_FT_32_6_BNComp.pdf')
    # eff_plot.savefig('AUCROC_FT_BNComp.pdf')
    eff_plot.show()

    # Acccuracy Plot
    eff_plot = plt.figure()
    eff_ax = eff_plot.add_subplot()
    eff_ax.grid(True)
    eff_ax.set_xlabel('BOPs')
    eff_ax.set_ylabel('Accuracy')
    eff_ax.set_xscale("symlog", linthresh=1e6)
    eff_ax.set_xlim(1e4, 1e7)
    eff_ax.set_ylim(0.6, 0.8)
    # eff_ax.set_title("AUC vs BOPS (HLS4ML Jet Tagging Model)")
    eff_ax.errorbar([int(key) for key in FT_kf[0]['32b']],
                [z['accuracy'] for z in FT_kf[0]['32b'].values()],
                [z['accuracy_err'] for z in FT_kf[0]['32b'].values()],
                label='32b FT w/ L1 Reg & BN', capsize=8,
                linestyle='solid', color='red')  # , marker='.',markersize=10,
    for x, y, mark in zip([int(key) for key in FT_kf[0]['32b']],
                          [z['accuracy'] for z in FT_kf[0]['32b'].values()], markers):
        eff_ax.plot(x, y, linestyle='solid', marker=mark, markersize=10, color='red', label='_nolegend_')

    eff_ax.errorbar([int(key) for key in FT_NoBN_kf[0]['32b']],
                [z['accuracy'] for z in FT_NoBN_kf[0]['32b'].values()],
                [z['accuracy_err'] for z in FT_NoBN_kf[0]['32b'].values()],
                label='32b FT No Batch Norm', capsize=8,
                linestyle='solid', color='blue')  # , marker='.',markersize=10,
    for x, y, mark in zip([int(key) for key in FT_NoBN_kf[0]['32b']],
                          [z['accuracy'] for z in FT_NoBN_kf[0]['32b'].values()], markers):
        eff_ax.plot(x, y, linestyle='none', marker=mark, markersize=10, color='blue', label='_nolegend_')

    eff_ax.errorbar([int(key) for key in FT_kf[0]['6b']],
                [z['accuracy'] for z in FT_kf[0]['6b'].values()],
                [z['accuracy_err'] for z in FT_kf[0]['6b'].values()],
                label='6b FT w/ L1 Reg & BN', capsize=8,
                linestyle='dotted', color='red')  # , marker='.',markersize=10,
    for x, y, mark in zip([int(key) for key in FT_kf[0]['6b']],
                          [z['accuracy'] for z in FT_kf[0]['6b'].values()], markers):
        eff_ax.plot(x, y, linestyle='none', marker=mark, markersize=10, color='red', label='_nolegend_')

    eff_ax.errorbar([int(key) for key in FT_NoBN_kf[0]['6b']],
                [z['accuracy'] for z in FT_NoBN_kf[0]['6b'].values()],
                [z['accuracy_err'] for z in FT_NoBN_kf[0]['6b'].values()],
                label='6b FT No Batch Norm', capsize=8,
                linestyle='dotted', color='blue')  # , marker='.',markersize=10,
    for x, y, mark in zip([int(key) for key in FT_NoBN_kf[0]['6b']],
                          [z['accuracy'] for z in FT_NoBN_kf[0]['6b'].values()], markers):
        eff_ax.plot(x, y, linestyle='none', marker=mark, markersize=10, color='blue', label='_nolegend_')
    #eff_ax.add_artist(
    eff_ax.legend(title='Fine tuning pruning', loc='lower right', framealpha=0.5)
    # eff_ax.add_artist(
    #    plt.legend(handles=marker_lines, title='Percent pruned (approx.)', loc='upper right', framealpha=0.5))
    eff_plot.savefig('ACC_FT_32_6_BNComp.pdf')
    # eff_plot.savefig('ACC_FT_BNComp.pdf')
    eff_plot.show()

    # Accuracy plot
    eff_plot = plt.figure()
    eff_ax = eff_plot.add_subplot()
    eff_ax.grid(True)
    eff_ax.set_xlabel('BOPs')
    eff_ax.set_ylabel('Accuracy')
    eff_ax.set_xscale("symlog", linthresh=1e6)
    eff_ax.set_xlim(1e4, 1e7)
    eff_ax.set_ylim(0.6, 0.8)
    #eff_ax.set_title("Accuracy vs BOPS (HLS4ML Jet Tagging Model)")
    eff_ax.errorbar([int(key) for key in FT_kf[0]['32b']],
                [z['accuracy'] for z in FT_kf[0]['32b'].values()],
                [z['accuracy_err'] for z in FT_kf[0]['32b'].values()],
                label='32b FT w/ L1 Reg & BN', capsize=8,
                linestyle='solid', color='red')  # , marker='.',markersize=10,
    for x, y, mark in zip([int(key) for key in FT_kf[0]['32b']],
                          [z['accuracy'] for z in FT_kf[0]['32b'].values()], markers):
        eff_ax.plot(x, y, linestyle='none', marker=mark, markersize=10, color='red', label='_nolegend_')

    eff_ax.errorbar([int(key) for key in FT_NoL1_kf[0]['32b']],
                    [z['accuracy'] for z in FT_NoL1_kf[0]['32b'].values()],
                    [z['accuracy_err'] for z in FT_NoL1_kf[0]['32b'].values()],
                label='32b FT No L1 Reg.', capsize=8,
                linestyle='solid', color='blue')  # , marker='.',markersize=10,
    for x, y, mark in zip([int(key) for key in FT_NoL1_kf[0]['32b']],
                          [z['accuracy'] for z in FT_NoL1_kf[0]['32b'].values()], markers):
        eff_ax.plot(x, y, linestyle='none', marker=mark, markersize=10, color='blue', label='_nolegend_')

    eff_ax.errorbar([int(key) for key in FT_kf[0]['6b']],
                    [z['accuracy'] for z in FT_kf[0]['6b'].values()],
                    [z['accuracy_err'] for z in FT_kf[0]['6b'].values()],
                label='6b FT w/ L1 Reg & BN', capsize=8,
                linestyle='dotted', color='red')  # , marker='.',markersize=10,
    for x, y, mark in zip([int(key) for key in FT_kf[0]['6b']],
                          [z['accuracy'] for z in FT_kf[0]['6b'].values()], markers):
        eff_ax.plot(x, y, linestyle='none', marker=mark, markersize=10, color='red', label='_nolegend_')

    eff_ax.errorbar([int(key) for key in FT_NoL1_kf[0]['6b']],
                    [z['accuracy'] for z in FT_NoL1_kf[0]['6b'].values()],
                    [z['accuracy_err'] for z in FT_NoL1_kf[0]['6b'].values()],
                label='6b FT No L1 Reg.', capsize=8,
                linestyle='dotted', color='blue')  # , marker='.',markersize=10,
    for x, y, mark in zip([int(key) for key in FT_NoL1_kf[0]['6b']],
                          [z['accuracy'] for z in FT_NoL1_kf[0]['6b'].values()], markers):
        eff_ax.plot(x, y, linestyle='none', marker=mark, markersize=10, color='blue', label='_nolegend_')
    eff_ax.legend(title='Fine tuning pruning', loc='lower right', framealpha=0.5)
    #eff_ax.add_artist(
    #    plt.legend(handles=marker_lines, title='Percent pruned (approx.)', loc='upper right', framealpha=0.5))
    eff_plot.savefig('ACC_FT_32_6_L1.pdf')
    # eff_plot.savefig('ACC_FT_BNcomp.pdf')
    eff_plot.show()



    # Efficiency plot
    eff_plot = plt.figure()
    eff_ax = eff_plot.add_subplot()
    eff_ax.grid(True)
    eff_ax.set_xlabel('BOPs')
    eff_ax.set_ylabel('Neural Efficiency')
    eff_ax.set_xscale("symlog", linthresh=1e6)
    eff_ax.set_xlim(1e4, 1e7)
    eff_ax.set_ylim(0, 0.5)
    #eff_ax.set_title("Efficiency vs BOPS (HLS4ML Jet Tagging Model)")
    eff_ax.plot([int(key) for key in FT_kf[0]['32b']], [z['net_efficiency'] for z in FT_kf[0]['32b'].values()],
                label='32b FT w/ Batch Norm',
                linestyle='solid', color='red')  # , marker='.',markersize=10,
    for x, y, mark in zip([int(key) for key in FT_kf[0]['32b']],
                          [z['net_efficiency'] for z in FT_kf[0]['32b'].values()], markers):
        eff_ax.plot(x, y, linestyle='solid', marker=mark, markersize=10, color='red', label='_nolegend_')

    eff_ax.plot([int(key) for key in FT_NoBN_kf[0]['32b']], [z['net_efficiency'] for z in FT_NoBN_kf[0]['32b'].values()],
                label='32b FT No Batch Norm',
                linestyle='solid', color='blue')  # , marker='.',markersize=10,
    for x, y, mark in zip([int(key) for key in FT_NoBN_kf[0]['32b']],
                          [z['net_efficiency'] for z in FT_NoBN_kf[0]['32b'].values()], markers):
        eff_ax.plot(x, y, linestyle='none', marker=mark, markersize=10, color='blue', label='_nolegend_')

    eff_ax.plot([int(key) for key in FT_kf[0]['6b']], [z['net_efficiency'] for z in FT_kf[0]['6b'].values()],
                label='6b FT w/ Batch Norm',
                linestyle='dotted', color='red')  # , marker='.',markersize=10,
    for x, y, mark in zip([int(key) for key in FT_kf[0]['6b']],
                          [z['net_efficiency'] for z in FT_kf[0]['6b'].values()], markers):
        eff_ax.plot(x, y, linestyle='none', marker=mark, markersize=10, color='red', label='_nolegend_')

    eff_ax.plot([int(key) for key in FT_NoBN_kf[0]['6b']], [z['net_efficiency'] for z in FT_NoBN_kf[0]['6b'].values()],
                label='6b FT No Batch Norm',
                linestyle='dotted', color='blue')  # , marker='.',markersize=10,
    for x, y, mark in zip([int(key) for key in FT_NoBN_kf[0]['6b']],
                          [z['net_efficiency'] for z in FT_NoBN_kf[0]['6b'].values()], markers):
        eff_ax.plot(x, y, linestyle='none', marker=mark, markersize=10, color='blue', label='_nolegend_')
    eff_ax.add_artist(
        plt.legend(handles=eff_ax.get_lines(), title='Fine tuning pruning', loc='lower left', framealpha=0.5))
    #eff_ax.add_artist(
    #    plt.legend(handles=marker_lines, title='Percent pruned (approx.)', loc='upper right', framealpha=0.5))
    eff_plot.savefig('Eff_FT_32_6.pdf')
    # eff_plot.savefig('Eff_FT_rand{}.pdf'.format(rand))
    eff_plot.show()

    # AUCROC Plot
    for precision, color in zip(precisions, colors):
        eff_plot = plt.figure()
        eff_ax = eff_plot.add_subplot()
        eff_ax.grid(True)
        eff_ax.set_xlabel('BOPs')
        eff_ax.set_ylabel('AUC')
        eff_ax.set_xscale("symlog", linthresh=1e6)
        eff_ax.set_xlim(1e4, 1e7)
        eff_ax.set_ylim(0.45, 1)
        #eff_ax.set_title("AUC vs BOPS - BO ({} HLS4ML Jet Tagging Model)".format(precision))
        eff_ax.plot([int(key) for key in FT_kf[0][precision]], [z['auc_roc'] for z in FT_kf[0][precision].values()],
                    label='{}-bit FT'.format(precision.rstrip('b')),
                    linestyle='solid', color=color)  # , marker='.',markersize=10,
        for x, y, mark in zip([int(key) for key in FT_kf[0][precision]],
                              [z['auc_roc'] for z in FT_kf[0][precision].values()], markers):
            eff_ax.plot(x, y, linestyle='none', marker=mark, markersize=10, color=color, label='_nolegend_')
        eff_ax.scatter([int(key) for key in BO[precision]], [z['auc_roc'] for z in BO[precision].values()],
                       label='{}-bit BO'.format(precision.rstrip('b')), color=color, s=200,
                       alpha=0.1)  # , marker='.'
        for txt, x, y in zip([str(z['dims']) for z in BO[precision].values()], [int(key) for key in BO[precision]],
                             [z['auc_roc'] for z in BO[precision].values()]):
            if txt == str(best[precision]):
                eff_ax.annotate(txt, (x, y), color='black', label='_nolegend_', size=12)
                eff_ax.scatter(x, y, marker="*", s=200, label='Best BO {} ({})'.format(precision, txt), color=color,
                               edgecolor='black', zorder=10)

        eff_ax.legend(loc='best', title="Optimization Techniques", framealpha=0.5)
        eff_plot.savefig('AUCROC_BOFT_{}.pdf'.format(precision))
        # eff_plot.savefig('AUCROC_FT_rand{}.pdf'.format(rand))
        eff_plot.show()

    # Accuracy plot
    for precision, color in zip(precisions, colors):
        eff_plot = plt.figure()
        eff_ax = eff_plot.add_subplot()
        eff_ax.grid(True)
        eff_ax.set_xlabel('BOPs')
        eff_ax.set_ylabel('Accuracy')
        eff_ax.set_xscale("symlog", linthresh=1e6)
        eff_ax.set_xlim(1e4, 1e7)
        eff_ax.set_ylim(0.6, 0.8)
        #eff_ax.set_title("Accuracy vs BOPS - BO ({} HLS4ML Jet Tagging Model)".format(precision))
        eff_ax.plot([int(key) for key in FT_kf[0][precision]], [z['accuracy'] for z in FT_kf[0][precision].values()],
                    label='{}-bit FT'.format(precision.rstrip('b')),
                    linestyle='solid', color=color)  # , marker='.',markersize=10,
        for x, y, mark in zip([int(key) for key in FT_kf[0][precision]],
                              [z['accuracy'] for z in FT_kf[0][precision].values()], markers):
            eff_ax.plot(x, y, linestyle='none', marker=mark, markersize=10, color=color, label='_nolegend_')
        eff_ax.scatter([int(key) for key in BO[precision]], [z['accuracy'] for z in BO[precision].values()],
                       label='{}-bit BO'.format(precision.rstrip('b')), color=color,s=200,
                       alpha=0.1)  # , marker='.'
        for txt, x, y in zip([str(z['dims']) for z in BO[precision].values()], [int(key) for key in BO[precision]],
                             [z['accuracy'] for z in BO[precision].values()]):
            if txt == str(best[precision]):
                eff_ax.annotate(txt, (x, y), color='black', label='_nolegend_', size=12)
                eff_ax.scatter(x, y, marker="*", s=200, label='Best BO {} ({})'.format(precision, txt), color=color,
                               edgecolor='black', zorder=10)
        eff_ax.legend(loc='best', title="Optimization Techniques", framealpha=0.5)
        eff_plot.savefig('ACC_BOFT_{}.pdf'.format(precision))
        # eff_plot.savefig('ACC_FT_rand{}.pdf'.format(rand))
        eff_plot.show()

    # Efficiency plot
    for precision, color in zip(precisions, colors):
        eff_plot = plt.figure()
        eff_ax = eff_plot.add_subplot()
        eff_ax.grid(True)
        eff_ax.set_xlabel('BOPs')
        eff_ax.set_ylabel('Efficiency')
        eff_ax.set_xscale("symlog", linthresh=1e6)
        eff_ax.set_xlim(1e4, 1e7)
        eff_ax.set_ylim(0, 0.6)
        #eff_ax.set_title("Efficiency vs BOPS BO ({} HLS4ML Jet Tagging Model)".format(precision))
        eff_ax.plot([int(key) for key in FT_kf[0][precision]], [z['net_efficiency'] for z in FT_kf[0][precision].values()],
                    label='{}-bit FT'.format(precision.rstrip('b')),
                    linestyle='solid', color=color)  # , marker='.',markersize=10,
        for x, y, mark in zip([int(key) for key in FT_kf[0][precision]],
                              [z['net_efficiency'] for z in FT_kf[0][precision].values()], markers):
            eff_ax.plot(x, y, linestyle='none', marker=mark, markersize=10, color=color, label='_nolegend_')
        eff_ax.scatter([int(key) for key in BO[precision]], [z['net_efficiency'] for z in BO[precision].values()],
                       label='{}-bit BO'.format(precision.rstrip('b')), color=color,s=200,
                       alpha=0.1)  # , marker='.',markersize=10,

        for txt, x, y in zip([str(z['dims']) for z in BO[precision].values()], [int(key) for key in BO[precision]],
                             [z['net_efficiency'] for z in BO[precision].values()]):
            if txt == str(best[precision]):
                eff_ax.annotate(txt, (x, y), color='black', label='_nolegend_', size=12)
                eff_ax.scatter(x, y, marker="*", s=200, label='Best BO {} ({})'.format(precision, txt), color=color,
                               edgecolor='black', zorder=10)

        eff_ax.legend(loc='best', title="Optimization Techniques", framealpha=0.5)
        eff_plot.savefig('Eff_BOFT_{}.pdf'.format(precision))
        # eff_plot.savefig('ACC_FT_rand{}.pdf'.format(rand))
        eff_plot.show()


    # AUC ROC plot
    eff_plot = plt.figure()
    eff_ax = eff_plot.add_subplot()
    eff_ax.grid(True)
    eff_ax.set_xlabel('BOPs')
    eff_ax.set_ylabel('AUC ROC')
    eff_ax.set_xscale("symlog", linthresh=1e6)
    eff_ax.set_xlim(1e4, 1e7)
    eff_ax.set_ylim(.45, 1)
    #eff_ax.set_title("AUC ROC vs BOPS (HLS4ML Jet Tagging Model)")
    eff_ax.plot([int(key) for key in FT_kf[0]['32b']], [z['auc_roc'] for z in FT_kf[0]['32b'].values()],
                label='32b FT',
                linestyle='solid', color='red')  # , marker='.',markersize=10,
    for x, y, mark in zip([int(key) for key in FT_kf[0]['32b']],
                          [z['auc_roc'] for z in FT_kf[0]['32b'].values()], markers):
        eff_ax.plot(x, y, linestyle='solid', marker=mark, markersize=10, color='red', label='_nolegend_')

    eff_ax.plot([int(key) for key in FT_kf[0]['6b']], [z['auc_roc'] for z in FT_kf[0]['6b'].values()],
                label='6b FT',
                linestyle='solid', color='blue')  # , marker='.',markersize=10,
    for x, y, mark in zip([int(key) for key in FT_kf[0]['6b']],
                          [z['auc_roc'] for z in FT_kf[0]['6b'].values()], markers):
        eff_ax.plot(x, y, linestyle='none', marker=mark, markersize=10, color='blue', label='_nolegend_')

    eff_ax.scatter([int(key) for key in BO['32b']], [z['auc_roc'] for z in BO['32b'].values()],
                   label='32b BO', color='red', s=200,
                   alpha=0.1)  # , marker='.',markersize=10,
    for txt, x, y in zip([str(z['dims']) for z in BO['32b'].values()], [int(key) for key in BO['32b']],
                         [z['auc_roc'] for z in BO['32b'].values()]):
        if txt == str(best['32b']):
            eff_ax.annotate(txt, (x, y), color='red', label='_nolegend_', size=12)
            eff_ax.scatter(x, y, marker="*", s=200, label='Best BO {} ({})'.format('32b', txt), color='red',
                           edgecolor='black', zorder=10)

    eff_ax.scatter([int(key) for key in BO['6b']], [z['auc_roc'] for z in BO['6b'].values()],
                   label='6b BO', color='blue', s=200,
                   alpha=0.1)  # , marker='.',markersize=10,
    for txt, x, y in zip([str(z['dims']) for z in BO['6b'].values()], [int(key) for key in BO['6b']],
                         [z['auc_roc'] for z in BO['6b'].values()]):
        if txt == str(best['6b']):
            eff_ax.annotate(txt, (x, y), color='blue', label='_nolegend_', size=12)
            eff_ax.scatter(x, y, marker="*", s=200, label='Best BO {} ({})'.format('6b', txt), color='blue',
                           edgecolor='black', zorder=10)
    eff_ax.legend(title='Optimization Technique', loc='best', framealpha=0.5)
    # eff_ax.add_artist(
    #    plt.legend(handles=marker_lines, title='Percent pruned (approx.)', loc='upper right', framealpha=0.5))
    eff_plot.savefig('AUCROC_BOFT_32_6.pdf')
    # eff_plot.savefig('ACC_FT_BNcomp.pdf')
    eff_plot.show()

    # Efficiency plot
    eff_plot = plt.figure()
    eff_ax = eff_plot.add_subplot()
    eff_ax.grid(True)
    eff_ax.set_xlabel('BOPs')
    eff_ax.set_ylabel('Neural Efficiency')
    eff_ax.set_xscale("symlog", linthresh=1e6)
    eff_ax.set_xlim(1e4, 1e7)
    eff_ax.set_ylim(0, 0.5)
    #eff_ax.set_title("Efficiency vs BOPS (HLS4ML Jet Tagging Model)")
    eff_ax.plot([int(key) for key in FT_kf[0]['32b']], [z['net_efficiency'] for z in FT_kf[0]['32b'].values()],
                label='32b FT',
                linestyle='solid', color='red')  # , marker='.',markersize=10,
    for x, y, mark in zip([int(key) for key in FT_kf[0]['32b']],
                          [z['net_efficiency'] for z in FT_kf[0]['32b'].values()], markers):
        eff_ax.plot(x, y, linestyle='solid', marker=mark, markersize=10, color='red', label='_nolegend_')

    eff_ax.plot([int(key) for key in FT_kf[0]['6b']], [z['net_efficiency'] for z in FT_kf[0]['6b'].values()],
                label='6b FT',
                linestyle='solid', color='blue')  # , marker='.',markersize=10,
    for x, y, mark in zip([int(key) for key in FT_kf[0]['6b']],
                          [z['net_efficiency'] for z in FT_kf[0]['6b'].values()], markers):
        eff_ax.plot(x, y, linestyle='none', marker=mark, markersize=10, color='blue', label='_nolegend_')

    eff_ax.scatter([int(key) for key in BO['32b']], [z['net_efficiency'] for z in BO['32b'].values()],
                   label='32b BO', color='red', s=200,
                   alpha=0.1)  # , marker='.',markersize=10,
    for txt, x, y in zip([str(z['dims']) for z in BO['32b'].values()], [int(key) for key in BO['32b']],
                         [z['net_efficiency'] for z in BO['32b'].values()]):
        if txt == str(best['32b']):
            eff_ax.annotate(txt, (x, y), color='red', label='_nolegend_', size=12)
            eff_ax.scatter(x, y, marker="*", s=200, label='Best BO {} ({})'.format('32b', txt), color='red',
                           edgecolor='black', zorder=10)

    eff_ax.scatter([int(key) for key in BO['6b']], [z['net_efficiency'] for z in BO['6b'].values()],
                   label='6b BO', color='blue' , s=200,
                   alpha=0.1)  # , marker='.',markersize=10,
    for txt, x, y in zip([str(z['dims']) for z in BO['6b'].values()], [int(key) for key in BO['6b']],
                         [z['net_efficiency'] for z in BO['6b'].values()]):
        if txt == str(best['6b']):
            eff_ax.annotate(txt, (x, y), color='blue', label='_nolegend_', size=12)
            eff_ax.scatter(x, y, marker="*", s=200, label='Best BO {} ({})'.format('6b', txt), color='blue',
                           edgecolor='black', zorder=10)
    eff_ax.legend(title='Optimization Technique', loc='best', framealpha=0.5)
    # eff_ax.add_artist(
    #    plt.legend(handles=marker_lines, title='Percent pruned (approx.)', loc='upper right', framealpha=0.5))
    eff_plot.savefig('Eff_BOFT_32_6.pdf')
    # eff_plot.savefig('Eff_FT_rand{}.pdf'.format(rand))
    eff_plot.show()




    #with open("../results_json/rand_trials_250e/BO/BO.json", "r") as read_file:
    #    BO = json.load(read_file)

    #best = {'4b':[44,32,32],
    #        '6b':[54,32,32],
    #        '12b':[64,32,19],
    #        '32b':[64,28,27]}

  #  labels_list = ['j_g', 'j_q', 'j_w', 'j_z', 'j_t']


   # # AUCROC Plot
   #  for precision, color in zip(precisions, colors):
   #      eff_plot = plt.figure()
   #      eff_ax = eff_plot.add_subplot()
   #      eff_ax.grid(True)
   #      eff_ax.set_xlabel('BOPs')
   #      eff_ax.set_ylabel('AUC')
   #      eff_ax.set_xscale("symlog", linthresh=1e6)
   #      eff_ax.set_xlim(1e4, 1e7)
   #      eff_ax.set_ylim(0.45, 1)
   #      #eff_ax.set_title("AUC vs BOPS - BO ({} HLS4ML Jet Tagging Model)".format(precision))
   #      eff_ax.scatter([int(key) for key in BO[precision]], [z['auc_roc'] for z in BO[precision].values()],
   #                  label='{}-bit'.format(precision.rstrip('b')), color=color, alpha = 0.5, s=200)  # , marker='.',markersize=10,
   #      for txt, x, y in zip([str(z['dims']) for z in BO[precision].values()], [int(key) for key in BO[precision]],
   #                                             [z['auc_roc'] for z in BO[precision].values()]):
   #          if txt == str(best[precision]):
   #              eff_ax.annotate(txt, (x, y),color='black', label='_nolegend_')
   #              eff_ax.scatter(x, y, marker="*", s=200, label='Best {} ({})'.format(precision,txt), color=color, edgecolor='black' )
   #      eff_ax.legend(loc='best', title="Bayesian Optimization", framealpha=0.5)
   #      eff_plot.savefig('AUCROC_BO_{}.pdf'.format(precision))
   #      # eff_plot.savefig('AUCROC_FT_rand{}.pdf'.format(rand))
   #      eff_plot.show()

 #    # Accuracy plot
 #    for precision, color in zip(precisions, colors):
 #        eff_plot = plt.figure()
 #        eff_ax = eff_plot.add_subplot()
 #        eff_ax.grid(True)
 #        eff_ax.set_xlabel('BOPs')
 #        eff_ax.set_ylabel('Accuracy')
 #        eff_ax.set_xscale("symlog", linthresh=1e6)
 #        eff_ax.set_xlim(1e4, 1e7)
 #        eff_ax.set_ylim(0.6, 0.8)
 #        #eff_ax.set_title("Accuracy vs BOPS - BO ({} HLS4ML Jet Tagging Model)".format(precision))
 #        eff_ax.scatter([int(key) for key in BO[precision]], [z['accuracy'] for z in BO[precision].values()],
 #                       label='{}-bit'.format(precision.rstrip('b')), color=color, alpha=0.5, s=200)  # , marker='.',markersize=10,
 #        for txt, x, y in zip([str(z['dims']) for z in BO[precision].values()], [int(key) for key in BO[precision]],
 #                                               [z['accuracy'] for z in BO[precision].values()]):
 #            if txt == str(best[precision]):
 #                eff_ax.annotate(txt, (x, y), color='black', label='_nolegend_')
 #                eff_ax.scatter(x, y, marker="*", s=200, label='Best {} ({})'.format(precision,txt), color=color, edgecolor='black' )
 #        eff_ax.legend(loc='best', title="Bayesian Optimization", framealpha=0.5)
 #        eff_plot.savefig('ACC_BO_{}.pdf'.format(precision))
 #        # eff_plot.savefig('ACC_FT_rand{}.pdf'.format(rand))
 #        eff_plot.show()
 #
 #    # Efficiency plot
 #    for precision, color in zip(precisions, colors):
 #        eff_plot = plt.figure()
 #        eff_ax = eff_plot.add_subplot()
 #        eff_ax.grid(True)
 #        eff_ax.set_xlabel('BOPs')
 #        eff_ax.set_ylabel('Efficiency')
 #        eff_ax.set_xscale("symlog", linthresh=1e6)
 #        eff_ax.set_xlim(1e4, 1e7)
 #        eff_ax.set_ylim(0, 0.6)
 #        #eff_ax.set_title("Efficiency vs BOPS BO ({} HLS4ML Jet Tagging Model)".format(precision))
 #        eff_ax.scatter([int(key) for key in BO[precision]], [z['net_efficiency'] for z in BO[precision].values()],
 #                       label='{}-bit'.format(precision.rstrip('b')), color=color, alpha = 0.5, s=200)  # , marker='.',markersize=10,
 #        for txt, x, y in zip([str(z['dims']) for z in BO[precision].values()], [int(key) for key in BO[precision]],
 # #                                              [z['net_efficiency'] for z in BO[precision].values()]):
#            if txt == str(best[precision]):
#                eff_ax.annotate(txt, (x, y),color='black', label='_nolegend_')
#                eff_ax.scatter(x, y, marker="*", s=200, label='Best {} ({})'.format(precision,txt), color=color, edgecolor='black' )
#        eff_ax.legend(loc='best', title="Bayesian Optimization", framealpha=0.5)
#        eff_plot.savefig('Eff_BO_{}.pdf'.format(precision))
#        # eff_plot.savefig('ACC_FT_rand{}.pdf'.format(rand))
#        eff_plot.show()



