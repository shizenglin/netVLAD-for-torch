# netVLAD-for-torch
Torch7 implementation of netVLAD, actionVLAD

This is a beta version of netVLAD implementation.
'''
#How to use?
require('./fun_netVLAD_layer.lua')
paths.dofile('./fun_netVLAD_layer.lua')

K=10
N=3
D=128

netVLAD_layer = create_netVLAD_layer(K,N,D)
model = nn.Sequential()
model:add(netVLAD_layer)
'''
![image](https://github.com/shamangary/netVLAD-for-torch/blob/master/netVLAD.png)
