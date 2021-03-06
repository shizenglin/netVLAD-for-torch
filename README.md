# netVLAD-for-torch
Torch7 implementation of netVLAD, actionVLAD

This is a beta version of netVLAD implementation.

***How to use***

*Step.1*
```
Download the file to current directory.
```

*Step.2 (Inside your .lua file)*
```
require('./fun_netVLAD_layer.lua')

K=10  --number of anchors
N=3   --number of features
D=128 --feature dimension

netVLAD_layer = create_netVLAD_layer(K,N,D)
model = nn.Sequential()
model:add(netVLAD_layer)
```
![image](https://github.com/shamangary/netVLAD-for-torch/blob/master/netVLAD.png)


***Ref.***
```
[1] R. Arandjelović, P. Gronat, A. Torii, T. Pajdla, J. Sivic. "NetVLAD: CNN architecture for weakly supervised place recognition", CVPR, 2016.
[2] Girdhar, Rohit and Ramanan, Deva and Gupta, Abhinav and Sivic, Josef and Russell, Bryan. "ActionVLAD: Learning spatio-temporal aggregation for action classification", CVPR, 2017.
```
