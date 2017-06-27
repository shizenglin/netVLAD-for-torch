require('./netVLAD_residual.lua')


-- Remember input size would be: B(batch size)xC(channel number)xD(feature dim)
-- The "channel number C" would be considered as "feature number N" since netVLAD treat feature maps as local descriptors.

function create_netVLAD_layer(anchor_num,feature_num,feature_dim,lr)

	-------------------------------------------------------------------------------------------------------------------
	--Parameters settting
	-------------------------------------------------------------------------------------------------------------------

	netVLAD_opt = {
	   K = anchor_num,
	   N = feature_num,
	   LR = lr,
	   D = feature_dim,
	   alpha = 1,
	   eps = 1e-2
	}

	-------------------------------------------------------------------------------------------------------------------
	--"netVLAD_layer "
	-------------------------------------------------------------------------------------------------------------------
	netVLAD_layer = nn.Sequential()

	divSumAlongK = nn.Sequential()
	ct1 = nn.ConcatTable()
	ct1_a = nn.Identity()
	ct1_b = nn.Sequential():add(nn.Sum(2)):add(nn.AddConstant(netVLAD_opt.eps,true)):add(nn.Power(-1)):add(nn.Replicate(netVLAD_opt.K,2))
	divSumAlongK:add(ct1:add(ct1_a):add(ct1_b)):add(nn.CMulTable())
	ct2 = nn.ConcatTable()
	ct2_a = nn.Identity()
	ct2_b = nn.Sequential():add(nn.Identity()):add(nn.Power(2)):add(nn.MulConstant(netVLAD_opt.alpha*(-1),true)):add(nn.Exp()):add(divSumAlongK):add(nn.Sum(3)):add(nn.Replicate(netVLAD_opt.N,3))
	-- somehow you have to add 'nn.Identity()'' at the beginning of ct2_b
	netVLAD_layer:add(nn.netVLAD_residual(netVLAD_opt.K,netVLAD_opt.N,netVLAD_opt.D,netVLAD_opt.LR))
	netVLAD_layer:add(ct2:add(ct2_a):add(ct2_b)):add(nn.CMulTable()):add(nn.Sum(3))

   	return netVLAD_layer
end