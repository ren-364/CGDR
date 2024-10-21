import torch
import argparse
import numpy as np
import dill
import time
from torch.optim import Adam
import os
import torch.nn.functional as F
from collections import defaultdict
from scipy.sparse import csr_matrix
from models import GAMENet
from util import llprint, multi_label_metric, ddi_rate_score, get_n_params, makeTorchAdj
import math
torch.manual_seed(1203)
np.random.seed(1203)

model_name = 'GAMENet'
resume_name = ''

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--eval', action='store_true', default=False, help="eval mode")
parser.add_argument('--model_name', type=str, default=model_name, help="model name")
parser.add_argument('--resume_path', type=str, default=resume_name, help='resume path')
parser.add_argument('--ddi', action='store_true', default=False, help="using ddi")

args = parser.parse_args()
model_name = args.model_name
resume_name = args.resume_path

def eval(model, data_eval, voc_size, epoch):
    # evaluate
    print('')
    model.eval()
    smm_record = []
    ja, prauc, avg_p, avg_r, avg_f1 = [[] for _ in range(5)]
    case_study = defaultdict(dict)
    med_cnt = 0
    visit_cnt = 0
    for step, input in enumerate(data_eval):
        y_gt = []
        y_pred = []
        y_pred_prob = []
        y_pred_label = []
        for adm_idx, adm in enumerate(input):

            target_output1 = model(input[:adm_idx+1])

            y_gt_tmp = np.zeros(voc_size[2])
            y_gt_tmp[adm[2]] = 1
            y_gt.append(y_gt_tmp)

            target_output1 = F.sigmoid(target_output1).detach().cpu().numpy()[0]
            y_pred_prob.append(target_output1)
            y_pred_tmp = target_output1.copy()
            y_pred_tmp[y_pred_tmp>=0.5] = 1
            y_pred_tmp[y_pred_tmp<0.5] = 0
            y_pred.append(y_pred_tmp)
            y_pred_label_tmp = np.where(y_pred_tmp == 1)[0]
            y_pred_label.append(sorted(y_pred_label_tmp))
            visit_cnt += 1
            med_cnt += len(y_pred_label_tmp)


        smm_record.append(y_pred_label)
        adm_ja, adm_prauc, adm_avg_p, adm_avg_r, adm_avg_f1 = multi_label_metric(np.array(y_gt), np.array(y_pred), np.array(y_pred_prob))
        case_study[adm_ja] = {'ja': adm_ja, 'patient': input, 'y_label': y_pred_label}

        ja.append(adm_ja)
        prauc.append(adm_prauc)
        avg_p.append(adm_avg_p)
        avg_r.append(adm_avg_r)
        avg_f1.append(adm_avg_f1)
        llprint('\rEval--Epoch: %d, Step: %d/%d' % (epoch, step, len(data_eval)))

    # ddi rate
    ddi_rate = ddi_rate_score(smm_record)

    llprint('\tDDI Rate: %.4f, Jaccard: %.4f,  PRAUC: %.4f, AVG_PRC: %.4f, AVG_RECALL: %.4f, AVG_F1: %.4f\n' % (
        ddi_rate, np.mean(ja), np.mean(prauc), np.mean(avg_p), np.mean(avg_r), np.mean(avg_f1)
    ))
    #dill.dump(obj=smm_record, file=open('../../data/gamenet_records.pkl', 'wb'))
    #dill.dump(case_study, open(os.path.join('../saved', model_name, 'case_study.pkl'), 'wb'))

    # print('avg med', med_cnt / visit_cnt)

    return ddi_rate, np.mean(ja), np.mean(prauc), np.mean(avg_p), np.mean(avg_r), np.mean(avg_f1)
def create_NPMI_matrices(voc_size, data, med_voc, diag_voc, pro_voc, yuzhi):
    Ndiag, Npro, Nmed= voc_size
    med_count_in_train = np.zeros(Nmed)
    diag_count_in_train = np.zeros(Ndiag)
    pro_count_in_train = np.zeros(Npro)
    #med2med = np.zeros((Nmed, Nmed))
    diag2med = np.zeros((Ndiag, Nmed))
    #med2diag = np.zeros((Nmed, Ndiag))
    pro2med = np.zeros((Npro, Nmed))

    n_visit = 0
    for p in data:
        for m in p:
            n_visit = n_visit + 1
            cur_diag, cur_pro, cur_med = m
            for cm in cur_med:
                med_count_in_train[cm] += 1
            for cd in cur_diag:
                diag2med[cd][cur_med] += 1
                diag_count_in_train[cd] += 1
            for cp in cur_pro:
                pro2med[cp][cur_med] += 1
                pro_count_in_train[cp] += 1

    for cd in diag_voc.idx2word:
        for cm in med_voc.idx2word:
            if diag2med[cd][cm] != 0:
                temp = diag2med[cd][cm] / (diag_count_in_train[cd]*med_count_in_train[cm]) * n_visit
                temp1 = diag2med[cd][cm] / n_visit
                diag2med[cd][cm] = -1 * math.log(temp) / math.log(temp1)
        diag2med[cd][diag2med[cd] >= yuzhi] = 1
        diag2med[cd][diag2med[cd] < yuzhi] = 0
    for cp in pro_voc.idx2word:
        for cm in med_voc.idx2word:
            if pro2med[cp][cm] != 0:
                temp = pro2med[cp][cm] / (pro_count_in_train[cp]*med_count_in_train[cm]) * n_visit
                temp1 = pro2med[cp][cm] / n_visit
                pro2med[cp][cm] = -1 * (math.log(temp) / math.log(temp1))
        pro2med[cp][pro2med[cp] >= yuzhi] = 1
        pro2med[cp][pro2med[cp] < yuzhi] = 0
    diag2med = torch.tensor(diag2med)
    pro2med = torch.tensor(pro2med)
    return diag2med, pro2med


def to_coo(a, lenx, leny):
    row, col = np.nonzero(a.numpy())
    values = a[row, col]
    csr_a = csr_matrix((values, (row, col)), shape=(lenx, leny))
    return csr_a.tocoo()

def lr_poly(base_lr, iter, max_iter, power, current_length):
    # ratio_length = 1 - (float(current_length) / 30)
    # iter = iter + ratio_length
    iter = iter + current_length
    if iter > max_iter:
        iter = iter % max_iter
    return base_lr * ((1 - float(iter) / max_iter) ** (power))#+ (float(current_length) / 30) ** (power))
    # return base_lr * (((1 - float(iter) / max_iter) ** (power))+ 0.1*((1 - (float(current_length) / 30) ** (power))))

def main():
    if not os.path.exists(os.path.join("", model_name)):
        os.makedirs(os.path.join("", model_name))

    data_path = '../data/records_final.pkl'
    voc_path = '../data/voc_final.pkl'

    ehr_adj_path = '../data/ehr_adj_final.pkl'
    ddi_adj_path = '../data/ddi_A_final.pkl'
    device = torch.device('cuda:0')

    ehr_adj = dill.load(open(ehr_adj_path, 'rb'))
    ddi_adj = dill.load(open(ddi_adj_path, 'rb'))
    data = dill.load(open(data_path, 'rb'))
    voc = dill.load(open(voc_path, 'rb'))
    diag_voc, pro_voc, med_voc = voc['diag_voc'], voc['pro_voc'], voc['med_voc']

    split_point = int(len(data) * 2 / 3)
    data_train = data[:split_point]
    eval_len = int(len(data[split_point:]) / 2)
    data_test = data[split_point:split_point + eval_len]
    data_eval = data[split_point+eval_len:]

    EPOCH = 40
    LR = 0.0002
    TEST = args.eval
    Neg_Loss = args.ddi
    DDI_IN_MEM = args.ddi
    TARGET_DDI = 0.05
    T = 0.5
    decay_weight = 0.85
    yuzhi = 0.04

    voc_size = (len(diag_voc.idx2word), len(pro_voc.idx2word), len(med_voc.idx2word))
    diag2med, pro2med = create_NPMI_matrices(voc_size, data_train, med_voc, diag_voc, pro_voc, yuzhi)
    dm_Hetergraph = to_coo(diag2med, voc_size[0], voc_size[2])
    pm_Hetergraph = to_coo(pro2med, voc_size[1], voc_size[2])
    #ddi_Hetergraph = to_coo(ddi2ddi, voc_size[2], voc_size[2])

    dm_Hetertensor = makeTorchAdj(dm_Hetergraph, dm_Hetergraph.shape[0], dm_Hetergraph.shape[1])
    pm_Hetertensor = makeTorchAdj(pm_Hetergraph, pm_Hetergraph.shape[0], pm_Hetergraph.shape[1])
    #ddi_Hetertensor = makeTorchAdj(ddi_Hetergraph, ddi_Hetergraph.shape[0], ddi_Hetergraph.shape[1])
    d_Hetertensor = dm_Hetertensor
    p_Hetertensor = pm_Hetertensor


    model = GAMENet(voc, voc_size, diag2med, ddi_adj, emb_dim=128, device=device, ddi_in_memory=DDI_IN_MEM, d_Hetertensor=d_Hetertensor, p_Hetertensor=p_Hetertensor)
    if TEST:
        model.load_state_dict(torch.load(open(resume_name, 'rb')))
    model.to(device=device)

    print('parameters', get_n_params(model))
    optimizer = Adam(list(model.parameters()), lr=LR)

    if TEST:
        eval(model, data_test, voc_size, 0)
    else:
        history = defaultdict(list)
        best_epoch = 0
        best_ja = 0
        for epoch in range(EPOCH):
            loss_record1 = []
            start_time = time.time()
            model.train()
            prediction_loss_cnt = 0
            neg_loss_cnt = 0
            for step, input in enumerate(data_train):
                for idx, adm in enumerate(input):
                    seq_input = input[:idx+1]
                    loss1_target = np.zeros((1, voc_size[2]))
                    loss1_target[:, adm[2]] = 1
                    loss3_target = np.full((1, voc_size[2]), -1)
                    for idx, item in enumerate(adm[2]):
                        loss3_target[0][idx] = item

                    target_output1, batch_neg_loss = model(seq_input)

                    loss1 = F.binary_cross_entropy_with_logits(target_output1, torch.FloatTensor(loss1_target).to(device))
                    loss3 = F.multilabel_margin_loss(F.sigmoid(target_output1), torch.LongTensor(loss3_target).to(device))
                    if Neg_Loss:
                        target_output1 = F.sigmoid(target_output1).detach().cpu().numpy()[0]
                        target_output1[target_output1 >= 0.5] = 1
                        target_output1[target_output1 < 0.5] = 0
                        y_label = np.where(target_output1 == 1)[0]
                        current_ddi_rate = ddi_rate_score([[y_label]])
                        if current_ddi_rate <= TARGET_DDI:
                            loss = 0.9 * loss1 + 0.01 * loss3
                            prediction_loss_cnt += 1
                        else:
                            rnd = np.exp((TARGET_DDI - current_ddi_rate)/T)
                            if np.random.rand(1) < rnd:
                                loss = batch_neg_loss
                                neg_loss_cnt += 1
                            else:
                                loss = 0.9 * loss1 + 0.01 * loss3
                                prediction_loss_cnt += 1
                    else:
                        loss = 0.9 * loss1 + 0.01 * loss3

                    optimizer.zero_grad()
                    loss.backward(retain_graph=True)
                    optimizer.step()

                    loss_record1.append(loss.item())

                llprint('\rTrain--Epoch: %d, Step: %d/%d, L_p cnt: %d, L_neg cnt: %d' % (epoch, step, len(data_train), prediction_loss_cnt, neg_loss_cnt))
            # annealing
            T *= decay_weight

            ddi_rate, ja, prauc, avg_p, avg_r, avg_f1 = eval(model, data_eval, voc_size, epoch)

            history['ja'].append(ja)
            history['ddi_rate'].append(ddi_rate)
            history['avg_p'].append(avg_p)
            history['avg_r'].append(avg_r)
            history['avg_f1'].append(avg_f1)
            history['prauc'].append(prauc)

            end_time = time.time()
            elapsed_time = (end_time - start_time) / 60
            llprint('\tEpoch: %d, Loss: %.4f, One Epoch Time: %.2fm, Appro Left Time: %.2fh\n' % (epoch,
                                                                                                np.mean(loss_record1),
                                                                                                elapsed_time,
                                                                                                elapsed_time * (
                                                                                                            EPOCH - epoch - 1)/60))

            torch.save(model.state_dict(), open(os.path.join('', model_name, 'Epoch_%d_JA_%.4f_DDI_%.4f.model' % (epoch, ja, ddi_rate)), 'wb'))
            print('')
            if epoch != 0 and best_ja < ja:
                best_epoch = epoch
                best_ja = ja


        dill.dump(history, open(os.path.join('', model_name, 'history.pkl'), 'wb'))

        # test
        torch.save(model.state_dict(), open(
            os.path.join('', model_name, 'final.model'), 'wb'))

        print('best_epoch:', best_epoch)


if __name__ == '__main__':
    main()
