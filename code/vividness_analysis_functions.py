# Stores all functions for the single experiment (within-subject) analyses
# testing the relationship between vividness and memory attributes

# Rose Cooper - September 2021

#########################################

# imports: ----------------------------- #
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
# -------------------------------------- #



def quality_check(my_data):
    # uses RTs and performance to check quality of data and 
    # mark subjects to exclude
    
    # set up an empty list to store subject IDs to exclude
    exclude_subs = []
    Nsubs = len(my_data.participant.unique())
    
    
    # a) RT # --------------------- #
    # calculate median RT across all types of memory test response:
    rt_cols = ['resp_ret_vividness.rt',
               'resp_gistmem1.rt','resp_gistmem2.rt','resp_gistmem3.rt',
               'resp_detailmem1.rt','resp_detailmem2.rt','resp_detailmem3.rt']
    rt_data = my_data[['participant'] + rt_cols].melt(id_vars=['participant']).groupby(['participant']).median().reset_index()

    # find any subjects <= .5s and add to exclude list (cannot respond for .25 before clock starts)
    ps = rt_data.loc[rt_data['value'] <= .5,'participant'].to_list()
    exclude_subs.extend(ps)
    print('\nNumber of subjects with median RT <= .75s --',len(ps),'out of',Nsubs,'subjects')

        
    # b) Same Key # --------------------- #
    # calculate the frequency with which each key was used
    key_cols = ['resp_ret_vividness.keys',
                'resp_gistmem1.keys','resp_gistmem2.keys','resp_gistmem3.keys',
                'resp_detailmem1.keys','resp_detailmem2.keys','resp_detailmem3.keys']
    key_data = my_data[['participant'] + key_cols].melt(id_vars=['participant'])
    key_data = key_data.groupby(['participant'])['value'].value_counts().unstack().reset_index()

    # now work out if any % of presses is > 75%
    key_cols = [1.0,2.0,3.0,4.0,5.0,6.0]
    key_data[key_cols] = (key_data[key_cols]/168)*100
    key_data = key_data[key_cols].max(axis=1).to_frame()
    key_data['participant'] = my_data['participant'].unique()

    ps = key_data.loc[key_data[0] > 75,'participant'].to_list()
    exclude_subs.extend(ps)
    print('\nNumber of subjects with consistent key presses (> 75% same key) --',len(ps),'out of',Nsubs,'subjects')

        
    # c) Gist Memory # --------------------- #
    gist = my_data[['participant',
                    'resp_gistmem1.corr','resp_gistmem2.corr','resp_gistmem3.corr']].melt(id_vars='participant')
    gist = gist.groupby(['participant']).mean().reset_index()

    # exclude? chance is 25%
    ps = gist.loc[gist['value'] <=.3,'participant'].to_list()
    exclude_subs.extend(ps)
    print('\nNumber of subjects with gist memory <= 30% --',len(ps),'out of',Nsubs,'subjects')
        
        
    # d) Detail Memory # --------------------- #
    detail = my_data[['participant',
                      'resp_detailmem1.corr','resp_detailmem2.corr','resp_detailmem3.corr']].melt(id_vars='participant')
    detail = detail.groupby(['participant']).mean().reset_index()

    # exclude? chance is 50%
    ps = detail.loc[detail['value'] <=.55,'participant'].to_list()
    exclude_subs.extend(ps)
    print('\nNumber of subjects with detail memory <= 55% --',len(ps),'out of',Nsubs,'subjects')
        
        
    # ***return full list of excluded subjects***
    exclude_subs = np.unique(exclude_subs).tolist()  #removing duplicates from above exclusions
    
    # remove from my_data
    if len(exclude_subs) > 0:
        print('\nRemoving subjects from my_data ....')
        # remove from df
        my_data = my_data[~my_data['participant'].isin(exclude_subs)]  
    
    return my_data, exclude_subs
# -------------------------------------- #



def quality_vividness(my_data, exclude_subs):
    # add additional check which isn't reflective of data quality
    # necessarily but allows us to remove subjects who don't
    # have a reasonable distribution of vividness responses

    viv_subjs_exclude = []

    # calculate the frequency with which each key was used
    key_data = my_data[['participant','resp_ret_vividness.keys']].melt(id_vars=['participant'])
    key_data = key_data.groupby(['participant'])['value'].value_counts().unstack().reset_index()

    # now work out if any % of presses is > 90% -- 24 total trials
    key_cols = [1.0,2.0,3.0,4.0,5.0,6.0]
    key_data[key_cols] = (key_data[key_cols]/24)*100
    key_data = key_data[key_cols].max(axis=1).to_frame()
    key_data['participant'] = my_data['participant'].unique()

    viv_subjs_exclude = key_data.loc[key_data[0] > 90,'participant'].to_list()

    print('Number of subjects with consistent vividness responses (> 90% same key) --',len(viv_subjs_exclude),'out of',len(key_data.index),'subjects')
    if len(viv_subjs_exclude) > 0:
        print('\nRemoving subjects from my_data ....')
        # remove from df
        my_data = my_data[~my_data['participant'].isin(viv_subjs_exclude)]   
        
    exclude_subs.extend(viv_subjs_exclude) #add to full list of excluded subjects
    exclude_subs = np.unique(exclude_subs).tolist()  #removing duplicates from above exclusion

    return my_data, exclude_subs
# -------------------------------------- #
        
        
        
def format_memory_data(my_data):
    # formats the data from questions into responses per feature:
    
    # rename columns for merging across questions:
    feature1 = my_data[['participant','event_id','q1_type','resp_gistmem1.corr','resp_detailmem1.corr']]
    feature1.columns=["participant","event","type","gist","detail"]

    feature2 = my_data[['participant','event_id','q2_type','resp_gistmem2.corr','resp_detailmem2.corr']]
    feature2.columns=["participant","event","type","gist","detail"]

    feature3 = my_data[['participant','event_id','q3_type','resp_gistmem3.corr','resp_detailmem3.corr']]
    feature3.columns=["participant","event","type","gist","detail"]

    # concatenate into long format
    feature_data = pd.concat([feature1,feature2,feature3], axis=0, ignore_index=True)
    
    
    # now convert to wide format and merge with vividness
    feature_wide = feature_data.pivot_table(index=['participant','event'],columns='type').reset_index()

    # add vividness ratings, by event, to feature_data
    memory_data = my_data[['participant','event_id','resp_ret_vividness.keys']]
    memory_data.columns=["participant","event","vividness"]
    memory_data = memory_data.merge(feature_wide, on=['participant','event'])

    # format column names so not tuples
    viv_columns = memory_data.columns[3:9]
    memory_data.columns = memory_data.columns[0:3].tolist() + ['_'.join(i) for i in viv_columns]

    return memory_data
# -------------------------------------- #



def vividness_correlations(vividness_data, features):
    # runs within-subject correlations between vividness
    # and (continuous) memory attributes
    # returns spearman r
    
    cors = pd.DataFrame(index=vividness_data['participant'].unique(),
                        columns=features)
    for p in vividness_data['participant'].unique():
        sub_data = vividness_data[vividness_data['participant'] == p]

        # correlate with vividness
        sub_cors = sub_data[["vividness"] + features].corr(method="spearman").loc["vividness",features].astype('float')
        cors_idx = (np.isnan(sub_cors)) | (np.round(np.abs(sub_cors),1) == 1)
        sub_cors[cors_idx] = 0
        cors.loc[p,features] = sub_cors

    return cors.reset_index()
# -------------------------------------- #



def fetch_sig_cors(subject_cors, group_cors, sym=True):
    # returns labels for significant correlations as an 
    # r x c matrix the same size as the input correlation matrix
    # * = p < .05 bonferroni-corrected
    # sym = True (symmetrical cor matrix)
    
    # get alpha (corrected):
    x_var = group_cors.shape[0]
    y_var = group_cors.shape[1]
    if sym:
        this_alpha = .05/(((x_var*y_var)-x_var)/2)
    else:
        this_alpha = .05/(x_var*y_var)
    
    # run t-tests:
    ps = stats.ttest_1samp(subject_cors, 0, axis=2, nan_policy='omit')[1]  #returns matrix of p values ([0] is the t values)
    np.fill_diagonal(ps, 1)
    
    # create mask
    p_mask = pd.DataFrame(ps < this_alpha)
    d = {True: '*', False: ''}
    p_mask = np.asarray(p_mask.replace(d))

    # now create annotations (concatenate asterix with str cor value)
    labels = (np.asarray(["{0}\n{1:.2f}".format(sig, value)
                          for sig, value in zip(p_mask.flatten(),
                                                np.array(group_cors).flatten())])
             ).reshape(x_var,y_var)

    return p_mask, labels
# -------------------------------------- #



def plot_cor_heatmap(cor_mat, l = True, cmap='magma', title=''):
    # l = labels 
    
    #mask to remove diagonal (if symmetrical)
    my_mask = np.zeros(cor_mat.shape)
    if cor_mat.shape[0] == cor_mat.shape[1]:
        np.fill_diagonal(my_mask, np.nan)
        
    if l == True:
        str_fmt = ".2f"
    else:
        str_fmt = "" #if input is pre-formatted labels
        
    # define figure size by dimensions:
    f_w = cor_mat.shape[1]*0.85
    f_h = cor_mat.shape[0]*0.75
           
    # colorbar range
    c_min = -0.1
    c_max = 0.4
    
    # heatmap
    plt.figure(figsize=(f_w,f_h), edgecolor="black")
    ax = sns.heatmap(cor_mat, mask=my_mask, 
                     cmap=cmap,vmin=c_min, vmax=c_max,
                     annot=l, annot_kws={"fontsize":14}, 
                     fmt=str_fmt, square=True)
    ax.set_facecolor('#e6e6e6')
    plt.xticks(fontsize=16, rotation=60, ha="right")
    plt.yticks(fontsize=16)
    plt.title(title, fontsize=24, y=1.05)

    # custom colorbar axis
    cax = plt.gcf().axes[-1]
    cax.tick_params(labelsize=14)
    cax.set_ylabel('Mean r', size=18, rotation=270, labelpad=15)
    
# -------------------------------------- #
