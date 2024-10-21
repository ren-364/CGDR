import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dnc import DNC
from layers import GraphConvolution
from util import MultiHeadAttention

'''
Our model
'''
class GCN(nn.Module):
    def __init__(self, voc_size, emb_dim, adj, device=torch.device('cpu:0')):
        super(GCN, self).__init__()
        self.voc_size = voc_size
        self.emb_dim = emb_dim
        self.device = device
        ddi_adj = self.normalize(adj + np.eye(adj.shape[0]))
        self.ddi_adj = torch.FloatTensor(ddi_adj).to(device)
        self.x = torch.eye(voc_size).to(device)
        self.gcn1 = GraphConvolution(voc_size, emb_dim)
        self.dropout = nn.Dropout(p=0.3)
        self.gcn3 = GraphConvolution(emb_dim, emb_dim)

    def forward(self):
        ddi_node_embedding = self.gcn1(self.x, self.ddi_adj)
        ddi_node_embedding = F.relu(ddi_node_embedding)
        ddi_node_embedding = self.dropout(ddi_node_embedding)
        ddi_node_embedding = self.gcn3(ddi_node_embedding, self.ddi_adj)
        return ddi_node_embedding

    def normalize(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = np.diagflat(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx


class CGMR(nn.Module):
    def __init__(self, voc, vocab_size, ddi_Hetertensor, diag2med, ddi_adj, ehr_adj, emb_dim, device=torch.device('cpu:0'), ddi_in_memory=True, d_Hetertensor=None, p_Hetertensor=None):
        super(CGMR, self).__init__()
        K = len(vocab_size)
        self.K = K
        self.vocab_size = vocab_size
        self.device = device
        self.tensor_ddi_adj = torch.FloatTensor(ddi_adj).to(device)
        self.ddi_in_memory = ddi_in_memory
        #self.embeddings = nn.ModuleList(
        #    [nn.Embedding(vocab_size[i], emb_dim) for i in range(K-1)])
        self.dropout = nn.Dropout(p=0.3)
        #self.gumbel_tau = 1
        self.diag2med = diag2med.to(device)
        self.MultiHead = nn.ModuleList([MultiHeadAttention(emb_dim * 2, num_heads=4, dropout=0.4) for _ in range(3)])
        #self.encoders = nn.ModuleList([mLSTM(emb_dim*2, emb_dim*2, 1, self.dropout) for _ in range(K-1)])

        self.query = nn.Sequential(
            nn.ReLU(),
            nn.Linear(emb_dim * 4, emb_dim * 2),
        )
        self.output = nn.Sequential(
            nn.ReLU(),
            nn.Linear(emb_dim * 10, emb_dim * 6),
            nn.ReLU(),
            nn.Linear(emb_dim * 6, vocab_size[2])
        )
        self.ehr_gcn = GCN(voc_size=vocab_size[2], emb_dim=emb_dim * 2, adj=ehr_adj, device=device)
        self.ddi_gcn = GCN(voc_size=vocab_size[2], emb_dim=emb_dim * 2, adj=ddi_adj, device=device)
        self.embeddings = nn.ModuleList(
            [nn.Embedding(vocab_size[i], emb_dim * 2) for i in range(K-1)])

        #self.d_Hetertensor = d_Hetertensor
        #self.p_Hetertensor = p_Hetertensor

        self.inter = nn.Parameter(torch.FloatTensor(1))
        #self.inter1 = nn.Parameter(torch.FloatTensor(1))
    def forward(self, input):
        #medEmd_dp, diagEmd, proEmd = self.embedding(self.d_Hetertensor, self.p_Hetertensor)
        #ddiEmd = self.ddi_gcn()
        medEmd = self.ehr_gcn() - self.ddi_gcn() * self.inter
        i1_seq = []
        i2_seq = []
        i3_seq = []
        def mean_embedding(embedding):
            return embedding.mean(dim=1).unsqueeze(dim=0)  # (1,1,dim)
        for adm in input:
            i1 = mean_embedding(self.dropout(self.embeddings[0](torch.LongTensor(adm[0]).unsqueeze(dim=0).to(self.device)))) # (1,1,dim)
            i2 = mean_embedding(self.dropout(self.embeddings[1](torch.LongTensor(adm[1]).unsqueeze(dim=0).to(self.device))))
            i3 = mean_embedding(medEmd[adm[2]].unsqueeze(dim=0))
            i1_seq.append(i1)
            i2_seq.append(i2)
            i3_seq.append(i3)
        i1_seq = torch.cat(i1_seq, dim=1) #(1,seq,dim)
        i2_seq = torch.cat(i2_seq, dim=1) #(1,seq,dim)
        i3_seq = torch.cat(i3_seq, dim=1)  # (1,seq,dim)

        cur_diag = np.zeros((1, self.vocab_size[0]))
        cur_diag[:, input[-1][0]] = 1
        prior_med_id = torch.nonzero(torch.mm(torch.tensor(cur_diag).to(self.device), self.diag2med))[:,-1]
        priorMed = mean_embedding(medEmd[prior_med_id].unsqueeze(dim=0) if len(prior_med_id) > 0 else torch.FloatTensor([0]).unsqueeze(dim=0).to(self.device))
        if len(input) > 1:
            i3_seq[:,:-1,:] = priorMed
        else:
            i3_seq = priorMed
        o1, diag_atten = self.MultiHead[0](i1_seq, i1_seq, i1_seq)
        o2, pro_atten = self.MultiHead[1](i2_seq, i2_seq, i2_seq)
        o3, med_atten = self.MultiHead[2](i3_seq, i3_seq, i3_seq)
        r1 = o1.squeeze(dim=0)[-1:]
        r2 = o2.squeeze(dim=0)[-1:]
        r3 = o3.squeeze(dim=0)[-1:]
        patient_representations = torch.cat([o1, o2], dim=-1).squeeze(dim=0) # (seq, dim*4)
        queries = self.query(patient_representations) # (seq, dim)

        # graph memory module
        '''I:generate current input'''
        query = queries[-1:] # (1,dim)

        '''G:generate graph memory bank and insert history information'''
        #ehr_embedding, ddi_embedding = self.gcn()
        drug_memory = medEmd

        if len(input) > 1:
            history_keys = queries[:(queries.size(0)-1)] # (seq-1, dim)

            history_values = np.zeros((len(input)-1, self.vocab_size[2]))
            for idx, adm in enumerate(input):
                if idx == len(input)-1:
                    break
                history_values[idx, adm[2]] = 1
            history_values = torch.FloatTensor(history_values).to(self.device) # (seq-1, size)

        '''O:read from global memory bank and dynamic memory bank'''
        key_weights1 = F.softmax(torch.mm(query, drug_memory.t()), dim=-1)  # (1, size)
        fact1 = torch.mm(key_weights1, drug_memory)  # (1, dim)

        med_weights = F.softmax(torch.mm(r3, drug_memory.t()), dim=-1)
        med_fact = torch.mm(med_weights, drug_memory)

        if len(input) > 1:
            visit_weight = F.softmax(torch.mm(query, history_keys.t())) # (1, seq-1)
            weighted_values = visit_weight.mm(history_values) # (1, size)
            fact2 = torch.mm(weighted_values, drug_memory) # (1, dim)
        else:
            fact2 = fact1
        '''R:convert O and predict'''
        output = self.output(torch.cat([r1, r2, fact1, fact2, med_fact], dim=-1)) # (1, dim)

        if self.training:
            neg_pred_prob = F.sigmoid(output)
            neg_pred_prob = neg_pred_prob.t() * neg_pred_prob  # (voc_size, voc_size)
            batch_neg = neg_pred_prob.mul(self.tensor_ddi_adj).mean()
            loss_ddi = 0.0005 * neg_pred_prob.mul(self.tensor_ddi_adj).sum()
            return output, batch_neg, loss_ddi
        else:
            return output


'''
DMNC
'''
class DMNC(nn.Module):
    def __init__(self, vocab_size, emb_dim=64, device=torch.device('cpu:0')):
        super(DMNC, self).__init__()
        K = len(vocab_size)
        self.K = K
        self.vocab_size = vocab_size
        self.device = device

        self.token_start = vocab_size[2]
        self.token_end = vocab_size[2] + 1

        self.embeddings = nn.ModuleList(
            [nn.Embedding(vocab_size[i] if i != 2 else vocab_size[2] + 2, emb_dim) for i in range(K)])
        self.dropout = nn.Dropout(p=0.3)

        self.encoders = nn.ModuleList([DNC(
            input_size=emb_dim,
            hidden_size=emb_dim,
            rnn_type='gru',
            num_layers=1,
            num_hidden_layers=1,
            nr_cells=16,
            cell_size=emb_dim,
            read_heads=1,
            batch_first=True,
            gpu_id=0,
            independent_linears=False
        ) for _ in range(K - 1)])

        self.decoder = nn.GRU(emb_dim + emb_dim * 2, emb_dim * 2,
                              batch_first=True)  # input: (y, r1, r2,) hidden: (hidden1, hidden2)
        self.interface_weighting = nn.Linear(emb_dim * 2, 2 * (emb_dim + 1 + 3))  # 2 read head (key, str, mode)
        self.decoder_r2o = nn.Linear(2 * emb_dim, emb_dim * 2)

        self.output = nn.Linear(emb_dim * 2, vocab_size[2] + 2)

    def forward(self, input, i1_state=None, i2_state=None, h_n=None, max_len=20):
        # input (3, code)
        i1_input_tensor = self.embeddings[0](
            torch.LongTensor(input[0]).unsqueeze(dim=0).to(self.device))  # (1, seq, codes)
        i2_input_tensor = self.embeddings[1](
            torch.LongTensor(input[1]).unsqueeze(dim=0).to(self.device))  # (1, seq, codes)

        o1, (ch1, m1, r1) = \
            self.encoders[0](i1_input_tensor, (None, None, None) if i1_state is None else i1_state)
        o2, (ch2, m2, r2) = \
            self.encoders[1](i2_input_tensor, (None, None, None) if i2_state is None else i2_state)

        # save memory state
        i1_state = (ch1, m1, r1)
        i2_state = (ch2, m2, r2)

        predict_sequence = [self.token_start] + input[2]
        if h_n is None:
            h_n = torch.cat([ch1[0], ch2[0]], dim=-1)

        output_logits = []
        r1 = r1.unsqueeze(dim=0)
        r2 = r2.unsqueeze(dim=0)

        if self.training:
            for item in predict_sequence:
                # teacher force predict drug
                item_tensor = self.embeddings[2](
                    torch.LongTensor([item]).unsqueeze(dim=0).to(self.device))  # (1, seq, codes)

                o3, h_n = self.decoder(torch.cat([item_tensor, r1, r2], dim=-1), h_n)
                read_keys, read_strengths, read_modes = self.decode_read_variable(h_n.squeeze(0))

                # read from i1_mem, i2_mem and i3_mem
                r1, _ = self.read_from_memory(self.encoders[0],
                                              read_keys[:, 0, :].unsqueeze(dim=1),
                                              read_strengths[:, 0].unsqueeze(dim=1),
                                              read_modes[:, 0, :].unsqueeze(dim=1), i1_state[1])

                r2, _ = self.read_from_memory(self.encoders[1],
                                              read_keys[:, 1, :].unsqueeze(dim=1),
                                              read_strengths[:, 1].unsqueeze(dim=1),
                                              read_modes[:, 1, :].unsqueeze(dim=1), i2_state[1])

                output = self.decoder_r2o(torch.cat([r1, r2], dim=-1))
                output = self.output(output + o3).squeeze(dim=0)
                output_logits.append(output)
        else:
            item_tensor = self.embeddings[2](
                torch.LongTensor([self.token_start]).unsqueeze(dim=0).to(self.device))  # (1, seq, codes)
            for idx in range(max_len):
                # predict
                # teacher force predict drug
                o3, h_n = self.decoder(torch.cat([item_tensor, r1, r2], dim=-1), h_n)
                read_keys, read_strengths, read_modes = self.decode_read_variable(h_n.squeeze(0))

                # read from i1_mem, i2_mem and i3_mem
                r1, _ = self.read_from_memory(self.encoders[0],
                                              read_keys[:, 0, :].unsqueeze(dim=1),
                                              read_strengths[:, 0].unsqueeze(dim=1),
                                              read_modes[:, 0, :].unsqueeze(dim=1), i1_state[1])

                r2, _ = self.read_from_memory(self.encoders[1],
                                              read_keys[:, 1, :].unsqueeze(dim=1),
                                              read_strengths[:, 1].unsqueeze(dim=1),
                                              read_modes[:, 1, :].unsqueeze(dim=1), i2_state[1])

                output = self.decoder_r2o(torch.cat([r1, r2], dim=-1))
                output = self.output(output + o3).squeeze(dim=0)
                output = F.softmax(output, dim=-1)
                output_logits.append(output)

                input_token = torch.argmax(output, dim=-1)
                input_token = input_token.item()
                item_tensor = self.embeddings[2](
                    torch.LongTensor([input_token]).unsqueeze(dim=0).to(self.device))  # (1, seq, codes)

        return torch.cat(output_logits, dim=0), i1_state, i2_state, h_n

    def read_from_memory(self, dnc, read_key, read_str, read_mode, m_hidden):
        read_vectors, hidden = dnc.memories[0].read(read_key, read_str, read_mode, m_hidden)
        return read_vectors, hidden

    def decode_read_variable(self, input):
        w = 64
        r = 2
        b = input.size(0)

        input = self.interface_weighting(input)
        # r read keys (b * w * r)
        read_keys = F.tanh(input[:, :r * w].contiguous().view(b, r, w))
        # r read strengths (b * r)
        read_strengths = F.softplus(input[:, r * w:r * w + r].contiguous().view(b, r))
        # read modes (b * 3*r)
        read_modes = F.softmax(input[:, (r * w + r):].contiguous().view(b, r, 3), -1)
        return read_keys, read_strengths, read_modes


'''
Leap
'''
class Leap(nn.Module):
    def __init__(self, voc_size, emb_dim=128, device=torch.device('cpu:0')):
        super(Leap, self).__init__()
        self.voc_size = voc_size
        self.device = device
        self.SOS_TOKEN = voc_size[2]
        self.END_TOKEN = voc_size[2]+1

        self.enc_embedding = nn.Sequential(
            nn.Embedding(voc_size[0], emb_dim, ),
            nn.Dropout(0.3)
        )
        self.dec_embedding = nn.Sequential(
            nn.Embedding(voc_size[2] + 2, emb_dim, ),
            nn.Dropout(0.3)
        )

        self.dec_gru = nn.GRU(emb_dim*2, emb_dim, batch_first=True)

        self.attn = nn.Linear(emb_dim*2, 1)

        self.output = nn.Linear(emb_dim, voc_size[2]+2)


    def forward(self, input, max_len=20):
        device = self.device
        # input (3, codes)
        input_tensor = torch.LongTensor(input[0]).to(device)
        # (len, dim)
        input_embedding = self.enc_embedding(input_tensor.unsqueeze(dim=0)).squeeze(dim=0)

        output_logits = []
        hidden_state = None
        if self.training:
            for med_code in [self.SOS_TOKEN] + input[2]:
                dec_input = torch.LongTensor([med_code]).unsqueeze(dim=0).to(device)
                dec_input = self.dec_embedding(dec_input).squeeze(dim=0) # (1,dim)

                if hidden_state is None:
                    hidden_state = dec_input

                hidden_state_repeat = hidden_state.repeat(input_embedding.size(0), 1) # (len, dim)
                combined_input = torch.cat([hidden_state_repeat, input_embedding], dim=-1) # (len, dim*2)
                attn_weight = F.softmax(self.attn(combined_input).t(), dim=-1) # (1, len)
                input_embedding = attn_weight.mm(input_embedding) # (1, dim)

                _, hidden_state = self.dec_gru(torch.cat([input_embedding, dec_input], dim=-1).unsqueeze(dim=0), hidden_state.unsqueeze(dim=0))
                hidden_state = hidden_state.squeeze(dim=0) # (1,dim)

                output_logits.append(self.output(F.relu(hidden_state)))

            return torch.cat(output_logits, dim=0)

        else:
            for di in range(max_len):
                if di == 0:
                    dec_input = torch.LongTensor([[self.SOS_TOKEN]]).to(device)
                dec_input = self.dec_embedding(dec_input).squeeze(dim=0) # (1,dim)
                if hidden_state is None:
                    hidden_state = dec_input
                hidden_state_repeat = hidden_state.repeat(input_embedding.size(0), 1)  # (len, dim)
                combined_input = torch.cat([hidden_state_repeat, input_embedding], dim=-1)  # (len, dim*2)
                attn_weight = F.softmax(self.attn(combined_input).t(), dim=-1)  # (1, len)
                input_embedding = attn_weight.mm(input_embedding)  # (1, dim)
                _, hidden_state = self.dec_gru(torch.cat([input_embedding, dec_input], dim=-1).unsqueeze(dim=0),
                                               hidden_state.unsqueeze(dim=0))
                hidden_state = hidden_state.squeeze(dim=0)  # (1,dim)
                output = self.output(F.relu(hidden_state))
                topv, topi = output.data.topk(1)
                output_logits.append(F.softmax(output, dim=-1))
                dec_input = topi.detach()
            return torch.cat(output_logits, dim=0)

'''
Retain
'''
class Retain(nn.Module):
    def __init__(self, voc_size, emb_size=64, device=torch.device('cpu:0')):
        super(Retain, self).__init__()
        self.device = device
        self.voc_size = voc_size
        self.emb_size = emb_size
        self.input_len = voc_size[0] + voc_size[1] + voc_size[2]
        self.output_len = voc_size[2]

        self.embedding = nn.Sequential(
            nn.Embedding(self.input_len + 1, self.emb_size, padding_idx=self.input_len),
            nn.Dropout(0.3)
        )

        self.alpha_gru = nn.GRU(emb_size, emb_size, batch_first=True)
        self.beta_gru = nn.GRU(emb_size, emb_size, batch_first=True)

        self.alpha_li = nn.Linear(emb_size, 1)
        self.beta_li = nn.Linear(emb_size, emb_size)

        self.output = nn.Linear(emb_size, self.output_len)

    def forward(self, input):
        device = self.device
        # input: (visit, 3, codes )
        max_len = max([(len(v[0]) + len(v[1]) + len(v[2])) for v in input])
        input_np = []
        for visit in input:
            input_tmp = []
            input_tmp.extend(visit[0])
            input_tmp.extend(list(np.array(visit[1]) + self.voc_size[0]))
            input_tmp.extend(list(np.array(visit[2]) + self.voc_size[0] + self.voc_size[1]))
            if len(input_tmp) < max_len:
                input_tmp.extend( [self.input_len]*(max_len - len(input_tmp)) )

            input_np.append(input_tmp)

        visit_emb = self.embedding(torch.LongTensor(input_np).to(device)) # (visit, max_len, emb)
        visit_emb = torch.sum(visit_emb, dim=1) # (visit, emb)

        g, _ = self.alpha_gru(visit_emb.unsqueeze(dim=0)) # g: (1, visit, emb)
        h, _ = self.beta_gru(visit_emb.unsqueeze(dim=0)) # h: (1, visit, emb)

        g = g.squeeze(dim=0) # (visit, emb)
        h = h.squeeze(dim=0) # (visit, emb)
        attn_g = F.softmax(self.alpha_li(g), dim=-1) # (visit, 1)
        attn_h = F.tanh(self.beta_li(h)) # (visit, emb)

        c = attn_g * attn_h * visit_emb # (visit, emb)
        c = torch.sum(c, dim=0).unsqueeze(dim=0) # (1, emb)

        return self.output(c)

'''
RF in train_LR.py
'''


