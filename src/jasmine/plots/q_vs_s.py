import matplotlib.pyplot as plt
import pandas as pd


sample_df = pd.read_csv(
    '/Users/stela/Documents/Scripts/orbital_task/data/gulls_orbital_motion_extracted/OMPLDG_croin_cassan.sample.csv')
sample_df['event_name'] = 'event_' + sample_df['SubRun'].astype(str) + '_' + sample_df['Field'].astype(str) + '_' + \
                          sample_df['EventID'].astype(str)

fig, ax = plt.subplots(figsize=(10*0.6, 6*0.6))

# Scatter plot
ax.scatter(sample_df['Planet_s'], sample_df['Planet_q'], alpha=0.7, edgecolor='k',
           s=10, label=f"Sample: # {len(sample_df['Planet_s'])}")

# Add labels
ax.set_ylabel('q: Mass Ratio', fontsize=12)
ax.set_xlabel('s: Separation', fontsize=12)
ax.set_title('Representative Sample', fontsize=14)
ax.set_xscale('log')
ax.set_yscale('log')
# plt.legend(loc='lower left')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.tight_layout()
plt.savefig('representative_sample.png', dpi=300)

plt.close(fig)

fig, ax = plt.subplots(figsize=(10.4*0.6, 6*0.6))

sample_df_chi2_filtered = sample_df[sample_df['ObsGroup_0_chi2'] > 500].copy()
# Scatter plot
ax.scatter(sample_df['Planet_s'], sample_df['Planet_q'], alpha=0.7, edgecolor='k',
           s=10, label=f"Sample: # {len(sample_df['Planet_s'])}")
ax.scatter(sample_df_chi2_filtered['Planet_s'], sample_df_chi2_filtered['Planet_q'], alpha=0.7, edgecolor='k',
           s=10, label=fr"$\Delta \chi^2 > 500$: # {len(sample_df_chi2_filtered['Planet_s'])}")

# Add labels
ax.set_ylabel('q: Mass Ratio', fontsize=12)
ax.set_xlabel('s: Separation', fontsize=12)
ax.set_title('Representative Sample', fontsize=14)
ax.set_xscale('log')
ax.set_yscale('log')
# plt.legend(loc='lower left')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.tight_layout()
plt.savefig('representative_sample_chi2_filtered.png', dpi=300)



