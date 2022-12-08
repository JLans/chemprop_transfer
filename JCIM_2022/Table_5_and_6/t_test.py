# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 13:16:57 2021

@author: joshua.l.lansford
"""
from scipy.stats import ttest_rel
#‘less’
Direct = [0.2638803348868579, 0.2715929791492503, 0.29171704145863
          , 0.25416680074791553, 0.29761974312489675]
M_10ani = [0.2408833 , 0.2691672 , 0.26903128, 0.25006907, 0.24574961]
M_10DFT = [0.27484476, 0.2795973 , 0.2767567 , 0.25828105, 0.25787767]

stat1, pvalue1 = ttest_rel(M_10DFT, Direct, alternative='less')
print(pvalue1)

stat1, pvalue2 = ttest_rel(M_10ani, M_10DFT, alternative='less')
print(pvalue2)

stat1, pvalue3 = ttest_rel(M_10ani, Direct, alternative='less')
print(pvalue3)

bs5_branched = [0.271007536, 0.270846702, 0.264717487]
bs25 = [0.285807809, 0.282437018, 0.283322045]

stat1, pvalue4 = ttest_rel(bs5_branched, bs25, alternative='less')
print(pvalue4)