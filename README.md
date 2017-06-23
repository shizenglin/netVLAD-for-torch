# netVLAD-for-torch
Torch7 implementation of netVLAD, actionVLAD

This is a beta version of netVLAD implementation.

##How to use##

#Step.1#
```
Download the file to current directory.
```

#Step.2 (Inside your .lua file)#
```
paths.dofile('./fun_netVLAD_layer.lua')

K=10  --number of dictionary, c
N=3   --number of features, x
D=128 --feature dimension

netVLAD_layer = create_netVLAD_layer(K,N,D)
model = nn.Sequential()
model:add(netVLAD_layer)
```
![image](https://github.com/shamangary/netVLAD-for-torch/blob/master/netVLAD.png)
