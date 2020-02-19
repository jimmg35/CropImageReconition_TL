import numpy as np 

ResNet = np.array([ [442,0,0,0] ,[6,323,0,0] ,[1,1,114,0] ,[3,0,1,84]])
VGG = np.array([ [442,0,0,0] ,[11,318,0,0] ,[107,9,0,0] ,[87,1,0,0]])
MobileNet = np.array([ [439,1,2,0] ,[1,326,1,1] ,[0,0,116,0] ,[5,3,0,80]])

def Kappa(Series):
    row, col = Series.shape
    po_A = 0
    po_B = 0

    for i in range(0,row):
        po_A += Series[i][i]

    for i in range(0,row):
        for j in range(0,col):
            po_B += Series[i][j]
            
    po = po_A/po_B
    
    pro = 0
    for i in range(0,row):
        row_N = 0
        col_N = 0
        for j in range(0,row):
            row_N += Series[i][j]
            col_N += Series[j][i]
        pro += row_N * col_N
    pe = pro / (po_B ** 2)

    kappa = (po - pe) / (1 - pe)
    return kappa



ResNet_kappa = Kappa(ResNet)
VGG_kappa = Kappa(VGG)
MobileNet_kappa = Kappa(MobileNet)

print(f'kappa of ResNet    : {ResNet_kappa}')
print(f'kappa of VGG       : {VGG_kappa}')
print(f'kappa of MobileNet : {MobileNet_kappa}')