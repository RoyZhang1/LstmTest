import numpy as np
import pdb
import sys
import os
import pickle
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as nnf
from torch.autograd import Variable
import pdb
import struct

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

class VN_Py_LSTM_Cell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(VN_Py_LSTM_Cell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.if_bias = bias
        # 
        self.fc_ih = nn.Linear(input_size, hidden_size*4)
        self.fc_hh = nn.Linear(hidden_size, hidden_size*4)
        # 
        self.h_perv = None
        self.c_perv = None
        pass

    def forward_test(self, inp, h_c_prev=None):
        if h_c_prev is None:
            h = self.h_perv
            c = self.c_perv
            if h is None:
                h = torch.zeros((inp.shape[0], self.hidden_size))
            if c is None:
                c = torch.zeros((inp.shape[0], self.hidden_size))
        else:
            h = h_c_prev[0]
            c = h_c_prev[1]
        # 
        assert(h.shape[0] == inp.shape[0])
        assert(c.shape[0] == inp.shape[0])

        h_t = torch.zeros((inp.shape[0], self.hidden_size))
        c_t = torch.zeros((inp.shape[0], self.hidden_size))

        x = inp.detach().numpy().astype(np.float32)
        h_ny = h.detach().numpy().astype(np.float32)
        c_ny = c.detach().numpy().astype(np.float32)

        wt_stride = self.hidden_size
        fc_ih_arr = self.fc_ih.weight.detach().numpy().astype(np.float32)
        wi = fc_ih_arr[0:wt_stride]
        wf = fc_ih_arr[wt_stride : 2 * wt_stride]
        wg = fc_ih_arr[2 * wt_stride : 3 * wt_stride]
        wo = fc_ih_arr[3 * wt_stride : ]

        bi_stride = self.hidden_size
        bi_ih_arr = self.fc_ih.bias.detach().numpy().astype(np.float32)
        bi = bi_ih_arr[0:bi_stride]
        bf = bi_ih_arr[bi_stride : 2 * bi_stride]
        bg = bi_ih_arr[2 * bi_stride : 3 * bi_stride]
        bo = bi_ih_arr[3 * bi_stride : ]

        wt_stride = self.hidden_size
        fc_hh_arr = self.fc_hh.weight.detach().numpy().astype(np.float32)
        wi_h = fc_hh_arr[0:wt_stride]
        wf_h = fc_hh_arr[wt_stride : 2 * wt_stride]
        wg_h = fc_hh_arr[2 * wt_stride : 3 * wt_stride]
        wo_h = fc_hh_arr[3 * wt_stride : ]

        bi_stride = self.hidden_size
        bi_hh_arr = self.fc_hh.bias.detach().numpy().astype(np.float32)
        bi_h = bi_hh_arr[0:bi_stride]
        bf_h = bi_hh_arr[bi_stride : 2 * bi_stride]
        bg_h = bi_hh_arr[2 * bi_stride : 3 * bi_stride]
        bo_h = bi_hh_arr[3 * bi_stride : ]

        for i in range(self.hidden_size):
            Iv = 0.0
            Fv = 0.0
            Gv = 0.0
            Ov = 0.0
            
            for k in range(self.input_size):
                Iv += x[0][k] * wi[i][k]
                Fv += x[0][k] * wf[i][k]
                Gv += x[0][k] * wg[i][k]
                Ov += x[0][k] * wo[i][k]
            
            Iv += bi[i]
            Fv += bf[i]
            Gv += bg[i]
            Ov += bo[i]

            for k in range(self.hidden_size):
                Iv += h_ny[0][k] * wi_h[i][k]
                Fv += h_ny[0][k] * wf_h[i][k]
                Gv += h_ny[0][k] * wg_h[i][k]
                Ov += h_ny[0][k] * wo_h[i][k]

            Iv += bi_h[i]
            Fv += bf_h[i]
            Gv += bg_h[i]
            Ov += bo_h[i]

            Iv = sigmoid(Iv)
            Fv = sigmoid(Fv)
            Gv = np.tanh(Gv)
            Ov = sigmoid(Ov)
            c_ = Fv * c_ny[0][i] + Iv * Gv
            h_ = Ov * np.tanh(c_)

            h_t[0][i] = h_
            c_t[0][i] = c_

            # 
        self.h_perv = h_t
        self.c_perv = c_t
        # 
        return h_t, c_t

    def loadParams(self, weight_ih, weight_hh, bias_ih, bias_hh):
        assert(self.fc_ih.weight.shape == weight_ih.shape)
        assert(self.fc_hh.weight.shape == weight_hh.shape)
        assert(self.fc_ih.bias.shape == bias_ih.shape)
        assert(self.fc_hh.bias.shape == bias_hh.shape)
        self.fc_ih.weight.data = weight_ih.data
        self.fc_hh.weight.data = weight_hh.data
        self.fc_ih.bias.data = bias_ih.data
        self.fc_hh.bias.data = bias_hh.data

    def forward(self, inp, h_c_prev=None):
        bTest = 0
        if bTest == 1:
            return forward_test(self, inp, h_c_prev)
        
        if h_c_prev is None:
            h = self.h_perv
            c = self.c_perv
            if h is None:
                h = torch.zeros((inp.shape[0], self.hidden_size))
            if c is None:
                c = torch.zeros((inp.shape[0], self.hidden_size))
        else:
            h = h_c_prev[0]
            c = h_c_prev[1]
        # 
        assert(h.shape[0] == inp.shape[0])
        assert(c.shape[0] == inp.shape[0])
        #
        ifgo_before_activation = self.fc_ih(inp) + self.fc_hh(h)
        ifgo_before_activation = ifgo_before_activation.view(inp.shape[0], 4, -1).permute(1, 0, 2)
        i = nnf.sigmoid(ifgo_before_activation[0])
        f = nnf.sigmoid(ifgo_before_activation[1])
        g = nnf.tanh(ifgo_before_activation[2])
        o = nnf.sigmoid(ifgo_before_activation[3])
        c_ = f * c + i * g # "*" in pytorch is default element-wise product.
        h_ = o * nnf.tanh(c_)
        # 
        self.h_perv = h_
        self.c_perv = c_
        # 
        return h_, c_

class VN_Py_LSTM_Multilayers_Block(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(VN_Py_LSTM_Multilayers_Block, self).__init__()
        pass
    
    def loadParams( self, 
                    weight_ih_lx_arr, 
                    weight_hh_lx_arr, 
                    bias_ih_lx_arr, 
                    bias_hh_lx_arr):
        pass
    
    def forward(self, seq_inputs, h_c_prev = None):
        return None, None

class VN_Py_LSTM_Bidirectional_Block(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(VN_Py_LSTM_Bidirectional_Block, self).__init__()
        pass
    
    def loadParams( self, 
                    weight_ih, 
                    weight_hh, 
                    bias_ih, 
                    bias_hh,
                    weight_ih_reverse, 
                    weight_hh_reverse, 
                    bias_ih_reverse, 
                    bias_hh_reverse ):
                   
        pass
    
    def forward(self, seq_inputs, h_c_prev = None):
        return None, None

class VN_Py_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, bidirectional = False, complex_rnn_type = 0):
        super(VN_Py_LSTM, self).__init__()
        # 
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.num_direction = 1
        self.bidirectional = bidirectional
        if bidirectional:
            self.num_direction = 2
        # 
        self.cell_stack_forward = nn.ModuleList()
        self.cell_stack_reverse = nn.ModuleList()
        # 
        self.complex_rnn_type = complex_rnn_type
        if complex_rnn_type == 0:
            for idx in range(num_layers):
                if idx == 0:
                    self.cell_stack_forward.append(VN_Py_LSTM_Cell(input_size, hidden_size, bias))
                else:
                    self.cell_stack_forward.append(VN_Py_LSTM_Cell(hidden_size * self.num_direction, hidden_size, bias))
            # 
            if bidirectional:
                for idx in range(num_layers):
                    if idx == 0:
                        self.cell_stack_reverse.append(VN_Py_LSTM_Cell(input_size, hidden_size, bias))
                    else:
                        self.cell_stack_reverse.append(VN_Py_LSTM_Cell(hidden_size * self.num_direction, hidden_size, bias))
        elif complex_rnn_type == 1:
            for idx in range(num_layers):
                if idx == 0:
                    self.cell_stack_forward.append(VN_Py_LSTM_Cell(input_size, hidden_size, bias))
                else:
                    self.cell_stack_forward.append(VN_Py_LSTM_Cell(hidden_size, hidden_size, bias))
            # 
            if bidirectional:
                for idx in range(num_layers):
                    if idx == 0:
                        self.cell_stack_reverse.append(VN_Py_LSTM_Cell(input_size, hidden_size, bias))
                    else:
                        self.cell_stack_reverse.append(VN_Py_LSTM_Cell(hidden_size, hidden_size, bias))
        else:
            print("Unsupported rnn type ", complex_rnn_type, ".")
            assert(0)

    def loadParams( self, 
                    weight_ih_lx_arr, 
                    weight_hh_lx_arr, 
                    bias_ih_lx_arr, 
                    bias_hh_lx_arr,
                    weight_ih_lx_reverse_arr = None, 
                    weight_hh_lx_reverse_arr = None, 
                    bias_ih_lx_reverse_arr = None, 
                    bias_hh_lx_reverse_arr = None
                    ):
        for idx in range(self.num_layers):
            self.cell_stack_forward[idx].loadParams(weight_ih_lx_arr[idx], weight_hh_lx_arr[idx], bias_ih_lx_arr[idx], bias_hh_lx_arr[idx])
            if self.bidirectional:
                self.cell_stack_reverse[idx].loadParams(weight_ih_lx_reverse_arr[idx], weight_hh_lx_reverse_arr[idx], bias_ih_lx_reverse_arr[idx], bias_hh_lx_reverse_arr[idx])

    def forward(self, seq_inputs, h_c_prev = None):
        
        seq_len = seq_inputs.shape[0]        
        batch = seq_inputs.shape[1]     
        # 
        seq_inps = []
        for idx_seq in range(seq_len):
            seq_inps.append(seq_inputs[idx_seq])
        #
        if self.bidirectional:
            if self.complex_rnn_type == 0: # type of pytorch is 0
                h0 = None
                c0 = None
                if h_c_prev is None:
                    h0 = torch.zeros(self.num_layers, self.num_direction, batch, self.hidden_size)
                    c0 = torch.zeros(self.num_layers, self.num_direction, batch, self.hidden_size)
                else:
                    h0 = h_c_prev[0].view(self.num_layers, self.num_direction, batch, self.hidden_size)
                    c0 = h_c_prev[1].view(self.num_layers, self.num_direction, batch, self.hidden_size)
                # 
                #forward
                seq_h_fw_arr = []   
                seq_c_fw_arr = []
                #backward
                seq_h_bw_arr = []
                seq_c_bw_arr = []
                for idx_layer in range(self.num_layers):
                    seq_h_fw = []
                    seq_c_fw = []
                    seq_h_bw = []
                    seq_c_bw = []
                    # 
                    seq_h_fw.append(h0[idx_layer][0])
                    seq_c_fw.append(c0[idx_layer][0])
                    seq_h_bw.append(h0[idx_layer][1])
                    seq_c_bw.append(c0[idx_layer][1])
                    # sequential forward and backward
                    for idx_seq in range(seq_len):
                        h_fw_curr, c_fw_curr = self.cell_stack_forward[idx_layer](seq_inps[idx_seq], (seq_h_fw[idx_seq], seq_c_fw[idx_seq]))
                        h_bw_curr, c_bw_curr = self.cell_stack_reverse[idx_layer](seq_inps[seq_len-1-idx_seq], (seq_h_bw[idx_seq], seq_c_bw[idx_seq]))
                        seq_h_fw.append(h_fw_curr)
                        seq_c_fw.append(c_fw_curr)
                        seq_h_bw.append(h_bw_curr)
                        seq_c_bw.append(c_bw_curr)
                    # concatenate the seq_h_fw and seq_h_bw together that get the new seq_inps of the next layer.
                    for idx_seq in range(seq_len):
                        seq_inps[idx_seq] = torch.cat((seq_h_fw[idx_seq+1], seq_h_bw[seq_len-idx_seq]), 1)
                    # 
                    seq_h_fw_arr.append(seq_h_fw)
                    seq_c_fw_arr.append(seq_c_fw)
                    seq_h_bw_arr.append(seq_h_bw)
                    seq_c_bw_arr.append(seq_c_bw)
                pass # endfor
                # get seq_outputs
                seq_outps_arr = []
                for idx_seq in range(seq_len):
                    seq_outps_arr.append(seq_inps[idx_seq].unsqueeze(0))
                seq_outps = torch.cat(seq_outps_arr, 0)
                # get h_n and c_n
                seq_h_arr = []
                seq_c_arr = []
                for idx_layer in range(self.num_layers):
                    seq_h_arr.append(torch.cat((seq_h_fw_arr[idx_layer][seq_len].unsqueeze(0), seq_h_bw_arr[idx_layer][seq_len].unsqueeze(0)), 0).unsqueeze(0))
                    seq_c_arr.append(torch.cat((seq_c_fw_arr[idx_layer][seq_len].unsqueeze(0), seq_c_bw_arr[idx_layer][seq_len].unsqueeze(0)), 0).unsqueeze(0))
                seq_h = torch.cat(seq_h_arr, 0).view(-1, batch, self.hidden_size)
                seq_c = torch.cat(seq_c_arr, 0).view(-1, batch, self.hidden_size)
                # 
                return seq_outps, (seq_h, seq_c)
            elif self.complex_rnn_type == 1: # other dnn frameworks may be 1
                return None, None
            else:
                print("Unsupported rnn type ", self.complex_rnn_type, "."); assert(0)
        else:
            h_prev_arr = []
            c_prev_arr = []
            if h_c_prev is None:
                for idx in range(self.num_layers):
                    h_prev_arr.append(torch.zeros(batch, self.hidden_size)) 
                    c_prev_arr.append(torch.zeros(batch, self.hidden_size))
            else:
                h0 = h_c_prev[0].view(self.num_layers, batch, self.hidden_size)
                c0 = h_c_prev[1].view(self.num_layers, batch, self.hidden_size)
                for idx in range(self.num_layers):
                    h_prev_arr.append(h0[idx]) 
                    c_prev_arr.append(c0[idx])
            # 
            seq_ouputs_arr = []
            for seq_idx in range(seq_len):
                for idx in range(self.num_layers):
                    if idx == 0:
                        h_prev_arr[idx], c_prev_arr[idx] = self.cell_stack_forward[idx](seq_inputs[seq_idx], (h_prev_arr[idx], c_prev_arr[idx]))
                    else:
                        h_prev_arr[idx], c_prev_arr[idx] = self.cell_stack_forward[idx](h_prev_arr[idx-1], (h_prev_arr[idx], c_prev_arr[idx]))
                seq_ouputs_arr.append(h_prev_arr[self.num_layers - 1].unsqueeze(0))
            # 
            for idx in range(self.num_layers):
                h_prev_arr[idx] = h_prev_arr[idx].unsqueeze(0).unsqueeze(0)
                c_prev_arr[idx] = c_prev_arr[idx].unsqueeze(0).unsqueeze(0)
            # 
            seq_outputs = torch.cat(seq_ouputs_arr, 0)
            h_curr = torch.cat(h_prev_arr, 0).view(self.num_layers, batch, self.hidden_size)
            c_curr = torch.cat(c_prev_arr, 0).view(self.num_layers, batch, self.hidden_size)
            return seq_outputs, (h_curr, c_curr)
        # 
        return None, None

def test_lstm_cell(input_size = 512, hidden_size = 256, bias = True):
    batch = 2
    seq_len = 10

    lstm_cell = nn.LSTMCell(input_size, hidden_size, bias)
    vn_lstm_cell = VN_Py_LSTM_Cell(input_size, hidden_size, bias)
    vn_lstm_cell.loadParams(lstm_cell.weight_ih, lstm_cell.weight_hh, lstm_cell.bias_ih, lstm_cell.bias_hh)
    
    inputs = torch.randn(seq_len, batch, input_size)
    h_arr = []
    c_arr = []
    h_arr.append(torch.zeros(batch, hidden_size)) 
    c_arr.append(torch.zeros(batch, hidden_size))
    
    outputs = []
    outputs_ = []

    for i in range(seq_len):
        hx, cx = lstm_cell(inputs[i], (h_arr[i], c_arr[i]))
        hx_, cx_ = vn_lstm_cell.forward(inputs[i])#, (h_arr[i], c_arr[i]))
        outputs.append(hx)
        outputs_.append(hx_)
        h_arr.append(hx)
        c_arr.append(cx)
        print(hx - hx_)

def readBytesFileToFloat(path):
    y = []
    with open(path, "rb+") as fb:
        x = fb.read()
        l = len(x)
        l = int(l / 4)
        for i in range(l):
            val = struct.unpack('f', x[i*4 : (i+1)*4])
            y.append(val)
    arr = np.array(y)
    return arr

def test_lstm(input_size = 512, hidden_size = 256, num_layers = 1, bias = True, bidirectional = True):
    batch = 1
    seq_len = 126
    num_directions = 1
    if(bidirectional):
        num_directions = 2
    # 
    lstm = nn.LSTM(input_size, hidden_size, num_layers, bias, bidirectional = bidirectional)
    vn_lstm = VN_Py_LSTM(input_size, hidden_size, num_layers, bias, bidirectional = bidirectional)
    # 
    weight_ih_lx_arr = []
    weight_hh_lx_arr = []
    bias_ih_lx_arr = []
    bias_hh_lx_arr = []
    weight_ih_lx_reverse_arr = []    
    weight_hh_lx_reverse_arr = []    
    bias_ih_lx_reverse_arr = []    
    bias_hh_lx_reverse_arr = []
    bSaveInputData = 1  #1 means save all LSTM input parameters from pytorch libarary 
    bReadbinFileTest = 1 # 0 means test lstm forward used pytorch linear, 1 means original implement

    #change to your dir
    Path = "/Users/zhangyao/code/test_lstm/num_layer_3/"

    for idx in range(num_layers):
        weight_ih_lx_arr.append(getattr(lstm, """weight_ih_l{}""".format(idx)))
        weight_hh_lx_arr.append(getattr(lstm, """weight_hh_l{}""".format(idx)))
        bias_ih_lx_arr.append(getattr(lstm, """bias_ih_l{}""".format(idx)))
        bias_hh_lx_arr.append(getattr(lstm, """bias_hh_l{}""".format(idx)))
        if bidirectional:
            weight_ih_lx_reverse_arr.append(getattr(lstm, """weight_ih_l{}_reverse""".format(idx)))
            weight_hh_lx_reverse_arr.append(getattr(lstm, """weight_hh_l{}_reverse""".format(idx)))
            bias_ih_lx_reverse_arr.append(getattr(lstm, """bias_ih_l{}_reverse""".format(idx)))
            bias_hh_lx_reverse_arr.append(getattr(lstm, """bias_hh_l{}_reverse""".format(idx)))
    
    inputs = torch.randn(seq_len, batch, input_size)
    h0 = torch.randn(num_layers * num_directions, batch, hidden_size)
    c0 = torch.randn(num_layers * num_directions, batch, hidden_size)

    output, (h1, c1) = lstm(inputs, (h0, c0))

    if bSaveInputData == 1:
        with open(Path + "inputs.bin", "wb+") as fb:
            fb.write(inputs.detach().numpy().astype(np.float32).tobytes())
        with open(Path + "h0.bin", "wb+") as fb:
            fb.write(h0.detach().numpy().astype(np.float32).tobytes())
        with open(Path + "c0.bin", "wb+") as fb:
            fb.write(c0.detach().numpy().astype(np.float32).tobytes())
        with open(Path + "output.bin", "wb+") as fb:
            fb.write(output.detach().numpy().astype(np.float32).tobytes())
        with open(Path + "h1.bin", "wb+") as fb:
            fb.write(h1.detach().numpy().astype(np.float32).tobytes())
        with open(Path + "c1.bin", "wb+") as fb:
            fb.write(c1.detach().numpy().astype(np.float32).tobytes())

        with open(Path + "weight_ih_l.bin", "wb+") as fb:
            fb.write(weight_ih_lx_arr[0].detach().numpy().astype(np.float32).tobytes())
        with open(Path + "weight_hh_l.bin", "wb+") as fb:
            fb.write(weight_hh_lx_arr[0].detach().numpy().astype(np.float32).tobytes())
        with open(Path + "bias_ih_l.bin", "wb+") as fb:
            fb.write(bias_ih_lx_arr[0].detach().numpy().astype(np.float32).tobytes())
        with open(Path + "bias_hh_l.bin", "wb+") as fb:
            fb.write(bias_hh_lx_arr[0].detach().numpy().astype(np.float32).tobytes())
        with open(Path + "weight_ih_l_reverse.bin", "wb+") as fb:
            fb.write(weight_ih_lx_reverse_arr[0].detach().numpy().astype(np.float32).tobytes())
        with open(Path + "weight_hh_l_reverse.bin", "wb+") as fb:
            fb.write(weight_hh_lx_reverse_arr[0].detach().numpy().astype(np.float32).tobytes())
        with open(Path + "bias_ih_l_reverse.bin", "wb+") as fb:
            fb.write(bias_ih_lx_reverse_arr[0].detach().numpy().astype(np.float32).tobytes())
        with open(Path + "bias_hh_l_reverse.bin", "wb+") as fb:
            fb.write(bias_hh_lx_reverse_arr[0].detach().numpy().astype(np.float32).tobytes())

    if bReadbinFileTest == 1:
        tmp = torch.from_numpy(readBytesFileToFloat(Path + "inputs.bin")).type(torch.FloatTensor)
        inputs_ = tmp.view(seq_len , 1, -1)
        tmp = torch.from_numpy(readBytesFileToFloat(Path + "h0.bin")).type(torch.FloatTensor)
        h0_ = tmp.view(2, 1, -1)
        tmp = torch.from_numpy(readBytesFileToFloat(Path + "c0.bin")).type(torch.FloatTensor)
        c0_ = tmp.view(2, 1, -1) 

        t_weight_ih_lx_arr = []
        t_weight_hh_lx_arr = []
        t_bias_ih_lx_arr = []
        t_bias_hh_lx_arr = []
        t_weight_ih_lx_reverse_arr = []    
        t_weight_hh_lx_reverse_arr = []    
        t_bias_ih_lx_reverse_arr = []    
        t_bias_hh_lx_reverse_arr = []

        tmp = torch.from_numpy(readBytesFileToFloat(Path + "weight_ih_l.bin")).type(torch.FloatTensor)
        t_weight_ih_lx_arr.append(tmp.view(hidden_size * 4, -1))
        tmp = torch.from_numpy(readBytesFileToFloat(Path + "weight_hh_l.bin")).type(torch.FloatTensor)
        t_weight_hh_lx_arr.append(tmp.view(hidden_size * 4, -1))
        tmp = torch.from_numpy(readBytesFileToFloat(Path + "bias_ih_l.bin")).type(torch.FloatTensor)
        t_bias_ih_lx_arr.append(tmp.view(hidden_size * 4))
        tmp = torch.from_numpy(readBytesFileToFloat(Path + "bias_hh_l.bin")).type(torch.FloatTensor)
        t_bias_hh_lx_arr.append(tmp.view(hidden_size * 4))
        tmp = torch.from_numpy(readBytesFileToFloat(Path + "weight_ih_l_reverse.bin")).type(torch.FloatTensor)
        t_weight_ih_lx_reverse_arr.append(tmp.view(hidden_size * 4, -1))
        tmp = torch.from_numpy(readBytesFileToFloat(Path + "weight_hh_l_reverse.bin")).type(torch.FloatTensor)
        t_weight_hh_lx_reverse_arr.append(tmp.view(hidden_size * 4, -1))
        tmp = torch.from_numpy(readBytesFileToFloat(Path + "bias_ih_l_reverse.bin")).type(torch.FloatTensor)
        t_bias_ih_lx_reverse_arr.append(tmp.view(hidden_size * 4))
        tmp = torch.from_numpy(readBytesFileToFloat(Path + "bias_hh_l_reverse.bin")).type(torch.FloatTensor)
        t_bias_hh_lx_reverse_arr.append(tmp.view(hidden_size * 4))
        
        vn_lstm.loadParams(t_weight_ih_lx_arr, t_weight_hh_lx_arr, t_bias_ih_lx_arr, t_bias_hh_lx_arr, t_weight_ih_lx_reverse_arr, t_weight_hh_lx_reverse_arr, t_bias_ih_lx_reverse_arr, t_bias_hh_lx_reverse_arr)
        output_, (h1_, c1_) = vn_lstm(inputs_, (h0_, c0_))     
    else:
        vn_lstm.loadParams(weight_ih_lx_arr, weight_hh_lx_arr, bias_ih_lx_arr, bias_hh_lx_arr, weight_ih_lx_reverse_arr, weight_hh_lx_reverse_arr, bias_ih_lx_reverse_arr, bias_hh_lx_reverse_arr)
        output_, (h1_, c1_) = vn_lstm(inputs, (h0, c0))

    pdb.set_trace()
    return True

# test_lstm_cell()
# test_pytorch_lstm()
ret = test_lstm()
assert(ret)





