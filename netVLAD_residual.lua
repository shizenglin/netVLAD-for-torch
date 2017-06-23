local netVLAD_residual, parent = torch.class('nn.netVLAD_residual', 'nn.Module')

function netVLAD_residual:__init(K,N,D,LR)
   parent.__init(self)
   
   self.LR = LR or 1e-4
   self.anchor_c = torch.Tensor(K,D):cuda()
   self.grad_anchor_c = torch.Tensor(K,D):zero():cuda()
   self.temp_grad_anchor_c = torch.Tensor(K,D):zero():cuda()

   self.m = nn.Sequential()
   local ct = nn.ConcatTable()
   local a = nn.Sequential():add(nn.SelectTable(1)):add(nn.Replicate(K,2))
   local b = nn.Sequential():add(nn.SelectTable(2)):add(nn.Replicate(N,3)):add(nn.MulConstant(-1,true))
   self.m:add(ct:add(a):add(b)):add(nn.CAddTable())
   self.m:cuda()
   
   self:reset(K,D)
end

function netVLAD_residual:reset(K,D)
   local stdv = 1./math.sqrt(K*D)
   self.anchor_c:uniform(-stdv,stdv)
end



function netVLAD_residual:parameters()
   return {self.anchor_c},{self.grad_anchor_c}
end


function netVLAD_residual:updateOutput(input)
   local nnre = nn.Replicate(input:size(1),1):cuda()
   local input_anchor_c = nnre:forward(self.anchor_c)
   output = self.m:forward({input,input_anchor_c})

   return output
end


function netVLAD_residual:updateGradInput(input, gradOutput)

   local nnre = nn.Replicate(input:size(1),1):cuda()
   local input_anchor_c = nnre:forward(self.anchor_c)
   local grad_temp = self.m:backward({input,input_anchor_c}, gradOutput)
   self.gradInput = grad_temp[1]
   self.temp_grad_anchor_c = grad_temp[2]:sum(1)
   return self.gradInput
end

function netVLAD_residual:accGradParameters(input, gradOutput, scale)
   scale = scale or 1
   scale = scale*self.LR
   self.grad_anchor_c:add(scale, self.temp_grad_anchor_c)
end

