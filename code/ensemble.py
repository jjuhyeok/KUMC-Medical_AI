import pandas as pd

submit_kmeans1 = pd.read_csv('submit_corrcovloss631_coatnet.csv')
submit_kmeans2 = pd.read_csv('submit_corrcovloss631_convnextv2_large.csv')
submit_kmeans3 = pd.read_csv('submit_corrcovloss631_xcit.csv')

submit_prefix1 = pd.read_csv('submit_prefix_coat.csv')
submit_prefix2 = pd.read_csv('submit_prefix_convnextv2.csv')
submit_prefix3 = pd.read_csv('submit_prefix_xcit.csv')


submit_kmeans1.iloc[:, 1:] = submit_kmeans1.iloc[:, 1:]*0.6 + submit_kmeans2.iloc[:, 1:]*0.2 + submit_kmeans3.iloc[:, 1:]*0.2
submit_prefix1.iloc[:, 1:] = submit_prefix1.iloc[:, 1:]*0.6 + submit_prefix2.iloc[:, 1:]*0.2 + submit_prefix3.iloc[:, 1:]*0.2


lst = submit_kmeans1.columns.to_list()
keyword = "AC"
AC_list = [item for item in lst if item.startswith(keyword)]
items_to_remove = ["ACAP3", "ACOT11", "ACTG2", "ACKR3", "ACAD11", "ACOT13", "ACTA2-AS1", "ACTA2", "ACER3", "ACACB", "ACYP1", "ACSBG1", "ACACA", "ACTG1", "ACP5", "ACTN4", "ACE2", "ACOT9"]
AC_list = [item for item in AC_list if item not in items_to_remove]

keyword = "AL"
AL_list = [item for item in lst if item.startswith(keyword)]
items_to_remove = ["ALG6", "ALCAM", "ALDH1L1", "ALDH1B1", "ALG2", "ALG10", "ALKBH2", "ALYREF", "ALKBH7"]
AL_list = [item for item in AL_list if item not in items_to_remove]

keyword = "LINC"
LINC_list = [item for item in lst if item.startswith(keyword)]

keyword = "SLC"
SLC_list = [item for item in lst if item.startswith(keyword)]

keyword = "ZNF"
ZNF_list = [item for item in lst if item.startswith(keyword)]

submit_kmeans1[AC_list] = submit_prefix1[AC_list]
submit_kmeans1[AL_list] = submit_prefix1[AL_list]
submit_kmeans1[LINC_list] = submit_prefix1[LINC_list]
submit_kmeans1[SLC_list] = submit_prefix1[SLC_list]
submit_kmeans1[ZNF_list] = submit_prefix1[ZNF_list]
submit_kmeans1.to_csv('final.csv', index=False)