import numpy as np

eip_in=np.array ([[1,0,1,0], [1,0,1,1], [0,1,0,1]], dtype=np.float64)
eip_out=np.array([1,1,0],dtype=np.float64)
iip_eip=np.random.rand(4,3)
eip_list=np.random.rand(1,3)
lr=np.random.rand (low=0,high=0.2, size=1)
mlbr_in=matrix_dot_product(eip_in,iip_eip)+eip_list
mlblr_out=sigmoid(mlblr_in)
eip_iip_in=matrix_dot_product(mlblr_out,eip_iip)+eip_dict
eip_iip=sigmoid(eip_iip_in)
eip=eip_out-eip_iip
iip_out=derivatives.sigmoid(eip_iip)
iip_in=derivatives.sigmoid(mlblr_out)
mlblr=eip * iip_out * lr
hleip=matrix_dot_product(mlblr,eip_iip.Transpose)
iip=hleip * iip_in
eip_iip=eip_iip+matrix_dot_product(mlblr_out.Transpose,mlblr) * lr
iip_eip=iip_eip+matrix_dot_product(eip_in.Transpose,iip) * lr
eip_list=eip_list+sum(iip,axis=0) * lr
eip_dict=eip_dict+sum(mlblr,axis=0) * lr

#### Summary of Variables substituted


##### X=eip_in
##### Y=eip_out
##### hidden_in=mlblr_in
##### hidden_act=mlblr_out
##### bh = eip_list
##### bout = eip_dict
##### wh = iip_eip
##### wout = eip_iip
##### wout_in = eip_iip_in
##### slope_out=iip_out
##### slope_hl=iip_in
##### E = eip
##### Ehl = hleip
##### delta_out=mlblr
##### delta_hl=iip
