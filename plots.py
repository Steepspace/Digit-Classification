#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df_nbd = pd.read_csv('results/nb_digits.csv')
df_nbf = pd.read_csv('results/nb_faces.csv')
df_pd = pd.read_csv('results/percp_digits.csv')
df_pf = pd.read_csv('results/percp_faces.csv')
df_md = pd.read_csv('results/mira_digits.csv')
df_mf = pd.read_csv('results/mira_faces.csv')

df = {}
df['digits'] = [df_nbd, df_pd, df_md]
df['faces'] = [df_nbf, df_pf, df_mf]

for i in df:
    for j in df[i]:
        j['accuracy_std'] /= np.sqrt(5)
        j.sort_values(by='n',inplace=True)

plt.figure(figsize=(20,15))
plt.subplot(221)
plt.plot(df['digits'][0]['n'], df['digits'][0]['time'],'ro--', ms=5, label='naive bayes')
plt.plot(df['digits'][1]['n'], df['digits'][1]['time'],'go--', ms=5, label='perceptron')
plt.plot(df['digits'][2]['n'], df['digits'][2]['time'],'bo--', ms=5, label='mira')

plt.tick_params(axis='both',direction='in',which='both')
plt.xlabel('n', size=15)
plt.ylabel('time [sec]', size=15)
plt.title('Classifier - Digits',size=15)

plt.xticks(np.arange(100, 1001, 100),fontsize=15)
plt.yticks(np.arange(0, 51, 5),fontsize=15)
# plt.xlim([50,1050])
# plt.ylim([0, 50])

plt.grid(which='major')
plt.legend(fontsize=15,labelcolor=['r','g','b']);

plt.subplot(222)

plt.plot(df['faces'][0]['n'], df['faces'][0]['time'],'ro--', ms=5, label='naive bayes')
plt.plot(df['faces'][1]['n'], df['faces'][1]['time'],'go--', ms=5, label='perceptron')
plt.plot(df['faces'][2]['n'], df['faces'][2]['time'],'bo--', ms=5, label='mira')

plt.tick_params(axis='both',direction='in',which='both')
plt.xlabel('n', size=15)
plt.ylabel('time [sec]', size=15)
plt.title('Classifier - Faces',size=15)

plt.xticks(np.arange(10, 101, 10),fontsize=15)
plt.yticks(fontsize=15)
# plt.xlim([0,150])
plt.ylim([0, 7])

plt.grid(which='major')
plt.legend(fontsize=15,labelcolor=['r','g','b']);

plt.subplot(223)

plt.errorbar(df['digits'][0]['n'], 100-df['digits'][0]['accuracy_mean'],yerr=df['digits'][0]['accuracy_std'],fmt='o',c='r',ls='--', capsize=5, ms=5, label='naive bayes')
plt.errorbar(df['digits'][1]['n'], 100-df['digits'][1]['accuracy_mean'],yerr=df['digits'][1]['accuracy_std'],fmt='o',c='g',ls='--', capsize=5, ms=5, label='perceptron')
plt.errorbar(df['digits'][2]['n'], 100-df['digits'][2]['accuracy_mean'],yerr=df['digits'][2]['accuracy_std'],fmt='o',c='b',ls='--', capsize=5, ms=5, label='mira')

plt.tick_params(axis='both',direction='in',which='both')
plt.xlabel('n', size=15)
plt.ylabel('prediction error [%]', size=15)
plt.title('Classifier - Digits',size=15)

plt.xticks(np.arange(100, 1001, 100),fontsize=15)
plt.yticks(fontsize=15)
# plt.xlim([0,150])
plt.ylim([15, 50])

plt.grid(which='major')
plt.legend(fontsize=15,labelcolor=['r','g','b']);

plt.subplot(224)

plt.errorbar(df['faces'][0]['n'], 100-df['faces'][0]['accuracy_mean'],yerr=df['faces'][0]['accuracy_std'],fmt='o',c='r',ls='--', capsize=5, ms=5, label='naive bayes')
plt.errorbar(df['faces'][1]['n'], 100-df['faces'][1]['accuracy_mean'],yerr=df['faces'][1]['accuracy_std'],fmt='o',c='g',ls='--', capsize=5, ms=5, label='perceptron')
plt.errorbar(df['faces'][2]['n'], 100-df['faces'][2]['accuracy_mean'],yerr=df['faces'][2]['accuracy_std'],fmt='o',c='b',ls='--', capsize=5, ms=5, label='mira')

plt.tick_params(axis='both',direction='in',which='both')
plt.xlabel('n', size=15)
plt.ylabel('prediction error [%]', size=15)
plt.title('Classifier - Faces',size=15)

plt.xticks(np.arange(10, 101, 10),fontsize=15)
plt.yticks(fontsize=15)
# plt.xlim([0,150])
plt.ylim([15, 50])

plt.grid(which='major')
plt.legend(fontsize=15,labelcolor=['r','g','b']);

plt.savefig('classifier.png', dpi=300)
