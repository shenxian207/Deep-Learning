"""
    Large-scale Point Cloud Semantic Segmentation with Superpoint Graphs
    http://arxiv.org/abs/1711.09869
    2017 Loic Landrieu, Martin Simonovsky
"""
from __future__ import division
from __future__ import print_function
from builtins import range

import torch
import torch.nn as nn
import torch.nn.functional as nnf
from torch.autograd import Variable
import ecc


class RNNGraphConvModule(nn.Module):
    """
    Computes recurrent graph convolution using filter weights obtained from a Filter generating network (`filter_net`).
    Its result is passed to RNN `cell` and the process is repeated over `nrepeats` iterations.
    Weight sharing over iterations is done both in RNN cell and in Filter generating network.
    """
    # #使用从滤波器生成网络（“filter_net”）获得的滤波器权重来计算循环图卷积。
    #  其结果传递给RNN“cell”，并在“nrepeats”迭代中重复该过程。
    #  迭代中的权重共享在RNN小区和过滤器生成网络中完成。
    def __init__(self, cell, filter_net, gc_info=None, nrepeats=1, cat_all=False, edge_mem_limit=1e20):
        super(RNNGraphConvModule, self).__init__()
        self._cell = cell
        self._isLSTM = 'LSTM' in type(cell).__name__
        self._fnet = filter_net
        self._nrepeats = nrepeats
        self._cat_all = cat_all
        self._edge_mem_limit = edge_mem_limit
        self.set_info(gc_info)

    def set_info(self, gc_info):
        self._gci = gc_info

    def forward(self, hx):
        # get graph structure information tensors
        #获得图形结构信息张量
        idxn, idxe, degs, degs_gpu, edgefeats = self._gci.get_buffers()
        edgefeats = Variable(edgefeats, requires_grad=False)

        # evalute and reshape filter weights (shared among RNN iterations)
        #评估和重塑滤波器权重（在RNN迭代之间共享）
        weights = self._fnet(edgefeats)
        nc = hx.size(1)
        assert hx.dim()==2 and weights.dim()==2 and weights.size(1) in [nc, nc*nc]  #断言，判定BUG
        if weights.size(1) != nc:
            weights = weights.view(-1, nc, nc)  #权重视图

        # repeatedly evaluate RNN cell
        #反复评估RNN细胞
        hxs = [hx]
        if self._isLSTM:
            cx = Variable(hx.data.new(hx.size()).fill_(0))

        for r in range(self._nrepeats):
            input = ecc.GraphConvFunction(nc, nc, idxn, idxe, degs, degs_gpu, self._edge_mem_limit)(hx, weights)
            if self._isLSTM:
                hx, cx = self._cell(input, (hx, cx))
            else:
                hx = self._cell(input, hx)
            hxs.append(hx)

        return torch.cat(hxs,1) if self._cat_all else hx


class ECC_CRFModule(nn.Module):
    """
    Adapted "Conditional Random Fields as Recurrent Neural Networks" (https://arxiv.org/abs/1502.03240)
    `propagation` should be ECC with Filter generating network producing 2D matrix.
    """
    #改编的“条件随机场作为回归神经网络”（https://arxiv.org/abs/1502.03240）
    # `propagation`应该是ECC，滤波器生成网络产生2D矩阵。
    def __init__(self, propagation, nrepeats=1):
        super(ECC_CRFModule, self).__init__()
        self._propagation = propagation
        self._nrepeats = nrepeats

    def forward(self, input):
        Q = nnf.softmax(input)
        for i in range(self._nrepeats):
            Q = self._propagation(Q) # todo: speedup possible by sharing computation of fnet
            Q = input - Q            # TODO： 通过共享fnet的计算实现加速
            if i < self._nrepeats-1:
                Q = nnf.softmax(Q) # last softmax will be part of cross-entropy loss
        return Q                   #最后的softmax是交叉熵的一部分


class GRUCellEx(nn.GRUCell):
    """ Usual GRU cell extended with layer normalization and input gate.
    """
    #通常的GRU单元扩展了层规范化和输入门
    #归一化nn.InstanceNorm1d函数，计算H*W的均值
    # 参数：‘ini’来自期望输入的特征数；eps为保证数值稳定性，给分母加上的值
    #affine布尔值，当为TRUE是，添加可学习的仿射变换参数；track_running_stats布尔值，当为TRUE，记录训练过程中的均值与方差
    def __init__(self, input_size, hidden_size, bias=True, layernorm=True, ingate=True):
        super(GRUCellEx, self).__init__(input_size, hidden_size, bias)
        self._layernorm = layernorm
        self._ingate = ingate
        if layernorm:
            self.add_module('ini', nn.InstanceNorm1d(1, eps=1e-5, affine=False, track_running_stats=True))
            self.add_module('inh', nn.InstanceNorm1d(1, eps=1e-5, affine=False, track_running_stats=True))
        if ingate:
            self.add_module('ig', nn.Linear(hidden_size, input_size, bias=True))

    def _normalize(self, gi, gh):
        if self._layernorm: # 输入和隐藏的层规范化,如在 https://arxiv.org/abs/1607.06450 (Layer Normalization层规范化)
            gi = self._modules['ini'](gi.unsqueeze(1)).squeeze(1)
            gh = self._modules['inh'](gh.unsqueeze(1)).squeeze(1)
        return gi, gh  #返回定义的gi、gh参数

    def forward(self, input, hidden):
        if self._ingate:
            input = nnf.sigmoid(self._modules['ig'](hidden)) * input   #sigmoid激活函数，非线性激活函数

        # GRUCell in https://github.com/pytorch/pytorch/blob/master/torch/nn/_functions/rnn.py extended with layer normalization
        #
        if input.is_cuda and torch._version_.split('.')[0]=='0' :
            gi = nnf.linear(input, self.weight_ih)
            gh = nnf.linear(hidden, self.weight_hh)
            gi, gh = self._normalize(gi, gh)
            state = torch.nn._functions.thnn.rnnFusedPointwise.GRUFused
            try: #pytorch >=0.3
                return state.apply(gi, gh, hidden) if self.bias_ih is None else state.apply(gi, gh, hidden, self.bias_ih, self.bias_hh)
            except: #pytorch <=0.2
                return state()(gi, gh, hidden) if self.bias_ih is None else state()(gi, gh, hidden, self.bias_ih, self.bias_hh)

        gi = nnf.linear(input, self.weight_ih, self.bias_ih)
        gh = nnf.linear(hidden, self.weight_hh, self.bias_hh)
        gi, gh = self._normalize(gi, gh)
        i_r, i_i, i_n = gi.chunk(3, 1)
        h_r, h_i, h_n = gh.chunk(3, 1)

        #调用激活函数
        resetgate = nnf.sigmoid(i_r + h_r)
        inputgate = nnf.sigmoid(i_i + h_i)
        newgate = nnf.tanh(i_n + resetgate * h_n)
        hy = newgate + inputgate * (hidden - newgate)
        return hy

    def __repr__(self):
        s = super(GRUCellEx, self).__repr__() + '('
        if self._ingate:
            s += 'ingate'
        if self._layernorm:
            s += ' layernorm'
        return s + ')'


class LSTMCellEx(nn.LSTMCell):
    """ Usual LSTM cell extended with layer normalization and input gate.
    """
    #通常的LSTM单元扩展了层规范化和输入门。
    #add_module(name,module)函数，将一个子模式module添加到当前模块,该模块可以使用给定的名称name作为属性访问
    #self表示类的实例
    def __init__(self, input_size, hidden_size, bias=True, layernorm=True, ingate=True):
        super(LSTMCellEx, self).__init__(input_size, hidden_size, bias)  #对父类的调用
        self._layernorm = layernorm
        self._ingate = ingate
        if layernorm:
            self.add_module('ini', nn.InstanceNorm1d(1, eps=1e-5, affine=False))
            self.add_module('inh', nn.InstanceNorm1d(1, eps=1e-5, affine=False))# nn.InstanceNorm1d没查到
        if ingate:
            self.add_module('ig', nn.Linear(hidden_size, input_size, bias=True))

    def _normalize(self, gi, gh):
        if self._layernorm: # 输入和隐藏的layernorm层规范化，在https://arxiv.org/abs/1607.06450 (Layer Normalization)
            gi = self._modules['ini'](gi.unsqueeze(1)).squeeze(1)
            gh = self._modules['inh'](gh.unsqueeze(1)).squeeze(1)#squeeze函数，将输入张量中的去除并返回
        return gi, gh

    def forward(self, input, hidden):
        if self._ingate:
            input = nnf.sigmoid(self._modules['ig'](hidden[0])) * input

        # GRUCell in https://github.com/pytorch/pytorch/blob/master/torch/nn/_functions/rnn.py extended with layer normalization
        if input.is_cuda:
            gi = nnf.linear(input, self.weight_ih)
            gh = nnf.linear(hidden[0], self.weight_hh)
            gi, gh = self._normalize(gi, gh)
            state = torch.nn._functions.thnn.rnnFusedPointwise.LSTMFused
            try: #pytorch >=0.3
                return state.apply(gi, gh, hidden[1]) if self.bias_ih is None else state.apply(gi, gh, hidden[1], self.bias_ih, self.bias_hh)
            except: #pytorch <=0.2
                return state()(gi, gh, hidden[1]) if self.bias_ih is None else state()(gi, gh, hidden[1], self.bias_ih, self.bias_hh)

        gi = nnf.linear(input, self.weight_ih, self.bias_ih)
        gh = nnf.linear(hidden[0], self.weight_hh, self.bias_hh)
        gi, gh = self._normalize(gi, gh)

        ingate, forgetgate, cellgate, outgate = (gi+gh).chunk(4, 1)
        ingate = nnf.sigmoid(ingate)
        forgetgate = nnf.sigmoid(forgetgate)#激活函数的调用        cellgate = nnf.tanh(cellgate)
        outgate = nnf.sigmoid(outgate)

        cy = (forgetgate * hidden[1]) + (ingate * cellgate)
        hy = outgate * nnf.tanh(cy)
        return hy, cy

    def __repr__(self):
        s = super(LSTMCellEx, self).__repr__() + '('
        if self._ingate:
            s += 'ingate'
        if self._layernorm:
            s += ' layernorm'
        return s + ')'
