import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

sorted_residues = [13, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 78, 136, 139, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 217, 218, 219, 249, 260, 261, 263, 265, 266, 267, 268, 269, 270, 271, 273, 301, 317, 321, 324, 362, 364, 365, 366, 367, 368, 369, 370, 371, 372, 374, 396, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 414, 415, 416, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 460, 461, 462, 463, 464, 465, 475, 478, 488, 489, 490, 491, 492, 493, 494, 495, 496, 497, 498, 499, 500, 501, 502, 503, 506, 507, 508, 509, 510, 511, 512, 515, 516, 518, 519, 520, 521, 522, 523, 524, 525, 526, 527, 528, 529, 530, 536, 537, 538, 539, 557, 558, 559, 560, 561, 579, 580, 581, 582, 583, 584, 585, 586, 587, 588, 589, 590, 591, 592, 621, 624, 625, 626, 627, 628, 629, 631, 655, 656, 657, 658, 659, 660, 661, 662, 663, 666, 667, 683, 688, 689, 690, 691, 692, 693, 694, 695, 696, 697, 698, 699, 700, 701, 702, 705, 708, 709, 710, 711, 712, 713, 714, 715, 716, 717, 718, 719, 720, 721, 722, 723, 724, 725, 726, 727, 728, 729, 730, 731, 732, 733, 734, 737, 761, 762, 763, 764, 765, 766, 767, 768, 769, 770, 771, 772, 773, 774, 775, 776, 777, 778, 779, 780, 781, 782, 783, 784, 802, 803, 804, 806, 807, 808, 809, 810, 812, 813, 816, 833, 834, 835, 836, 837, 838, 839, 840, 842, 843, 844, 845, 846, 847, 848, 849, 850, 851, 852, 853, 854, 855, 858, 859, 860, 861, 862, 863, 864, 866, 867, 868, 869, 870, 893, 895, 896, 908, 910, 913, 916, 917, 918, 919, 920, 921, 922, 923, 924, 925, 926, 927, 928, 929, 930, 931, 932, 933, 934, 936, 937, 938, 940, 941, 948, 949, 951, 955, 956, 957, 958, 959, 960, 961, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010, 1011, 1012, 1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023, 1024, 1025, 1026, 1027, 1028, 1029, 1030, 1031, 1032, 1033, 1034, 1035, 1036, 1038, 1039, 1104, 1105, 1106, 1107, 1108, 1109, 1110, 1111, 1122, 1134, 1135, 1136, 1138]


cmut_tbl = pd.read_csv('data/cas9_nucleotide_distances.csv', index_col='Unnamed: 0')
cmut_tbl['dist'] = np.sqrt(cmut_tbl['dist**2'])
cmut_only_tbl = cmut_tbl[cmut_tbl['cmut'] != 'TM']
cmut_only_tbl


dist_df = cmut_only_tbl.groupby(['chain_pos_y','chain_x','chain_pos_x']).agg(list).reset_index()

temp = pd.DataFrame(cmut_only_tbl.groupby(['chain_pos_y','chain_x','chain_pos_x']).size()).reset_index().rename(columns={0: 'count'})

res_idx_map = {x: i for i, x in enumerate(sorted_residues)}

# compute total number of counts
pos_counts = np.zeros((388,2,20))
for _, row in temp.iterrows():
  if row['chain_x'] == 'B':
    pos_counts[res_idx_map[row['chain_pos_y']], 1, row['chain_pos_x']-1] += row['count']
  else: # if row['chain_x'] == 'C'
    pos_counts[res_idx_map[row['chain_pos_y']], 0, 30-row['chain_pos_x']] += row['count']

aa_indices = [730, 837,
              731, 838,
              732, 839,
              733, 1015,
              734, 1016]

fig, axn = plt.subplots(5, 2, sharey=True,
                          figsize=(12, 0.5*len(aa_indices))) # , sharex=True
cbar_ax = fig.add_axes([.91, .3, .03, .4])
for i, ax in enumerate(axn.flat):
    sns.heatmap(pos_counts[res_idx_map[aa_indices[i]]], ax=ax,
                cbar=i == 0,
                xticklabels=np.flip(np.arange(1,21)),
                yticklabels=['tsDNA','sgRNA'],
                vmin=0, vmax=672,
                cbar_ax=None if i else cbar_ax)# , cbar_ax=None if i else cbar_ax
    ax.set_ylabel("Residue\n"+str(aa_indices[i]))
fig.tight_layout(rect=[0, 0, .9, 1])
fig.savefig("out/fig4a.pdf")
fig.show()
