from __future__ import print_function
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import random
import os
import argparse
import numpy as np
from PreResNet import *
from sklearn.mixture import GaussianMixture
import dataloader_cifar as dataloader
import matplotlib.pyplot as plt
import copy
import seaborn as sns
# from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.cluster import Birch
import matplotlib

parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
parser.add_argument('--batch_size', default=128, type=int, help='train batchsize')
parser.add_argument('--lr', '--learning_rate', default=0.02, type=float, help='initial learning rate')
parser.add_argument('--noise_mode', default='asym')
parser.add_argument('--alpha', default=4, type=float, help='parameter for Beta')
parser.add_argument('--lambda_u', default=150, type=float, help='weight for unsupervised loss')
parser.add_argument('--p_threshold', default=0.5, type=float, help='clean probability threshold')
parser.add_argument('--T', default=0.5, type=float, help='sharpening temperature')
parser.add_argument('--num_epochs', default=300, type=int)
parser.add_argument('--r', default=0.3, type=float, help='noise ratio')
parser.add_argument('--id', default='')
parser.add_argument('--seed', default=123)
parser.add_argument('--gpuid', default=0, type=int)
parser.add_argument('--num_class', default=100, type=int)
# parser.add_argument('--data_path', default='./data/cifar-10-batches-py', type=str, help='path to dataset')
# parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--data_path', default='./data/cifar-100-python', type=str, help='path to dataset')
parser.add_argument('--dataset', default='cifar100', type=str)
args = parser.parse_args()

torch.cuda.set_device(args.gpuid)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

mse = torch.nn.MSELoss(reduction='none').cuda()


# Training
def train(epoch, net, net2, optimizer, labeled_trainloader, unlabeled_trainloader, mask=None, f_G=None, new_y=None):
    net.train()
    net2.eval()  # fix one network and train the other

    unlabeled_train_iter = iter(unlabeled_trainloader)
    num_iter = (len(labeled_trainloader.dataset) // args.batch_size) + 1
    mse_total = 0
    for batch_idx, (inputs_x, inputs_x2, labels_x, w_x) in enumerate(labeled_trainloader):
        try:
            inputs_u, inputs_u2 = unlabeled_train_iter.__next__()
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            inputs_u, inputs_u2 = unlabeled_train_iter.__next__()
        batch_size = inputs_x.size(0)

        # Transform label to one-hot，转为0-1矩阵
        labels_x = torch.zeros(batch_size, args.num_class).scatter_(1, labels_x.view(-1, 1), 1)
        w_x = w_x.view(-1, 1).type(torch.FloatTensor)

        inputs_x, inputs_x2, labels_x, w_x = inputs_x.cuda(), inputs_x2.cuda(), labels_x.cuda(), w_x.cuda()
        inputs_u, inputs_u2 = inputs_u.cuda(), inputs_u2.cuda()

        with torch.no_grad():
            # label co-guessing of unlabeled samples
            outputs_u11, feat_u11 = net(inputs_u, feat_out=True)
            outputs_u12, feat_u12 = net(inputs_u2, feat_out=True)
            outputs_u21, feat_u21 = net2(inputs_u, feat_out=True)
            outputs_u22, feat_u22 = net2(inputs_u2, feat_out=True)

            # 取average of 所有网络的输出，作者利用了所谓的augmentation
            pu = (torch.softmax(outputs_u11, dim=1) + torch.softmax(outputs_u12, dim=1)
                  + torch.softmax(outputs_u21, dim=1) + torch.softmax(outputs_u22, dim=1)) / 4
            ptu = pu ** (1 / args.T)  # temparature sharpening

            # Algorithm 1 中的shapen(qb,T)
            targets_u = ptu / ptu.sum(dim=1, keepdim=True)  # normalize
            targets_u = targets_u.detach()

            # label refinement of labeled samples
            outputs_x, feat_x1 = net(inputs_x, feat_out=True)
            outputs_x2, feat_x2 = net(inputs_x2, feat_out=True)

            # 取labeled的输出平均值
            px = (torch.softmax(outputs_x, dim=1) + torch.softmax(outputs_x2, dim=1)) / 2

            # 公式(3)(4)退火
            px = w_x * labels_x + (1 - w_x) * px
            ptx = px ** (1 / args.T)  # temparature sharpening

            targets_x = ptx / ptx.sum(dim=1, keepdim=True)  # normalize
            targets_x = targets_x.detach()
            # aaa = torch.argmax(labels_x, dim=1)
            # mse_loss = torch.sum(mse((feat_x1+feat_x2)/2, f_G[aaa]), 1)
            # mse_total = (mse_total + torch.sum(mse_loss) / len(mse_loss))/2
        # mixmatch
        l = np.random.beta(args.alpha, args.alpha)
        # 促使X'更加靠近labeled sample而不是无监督样本
        l = max(l, 1 - l)

        all_inputs = torch.cat([inputs_x, inputs_x2, inputs_u, inputs_u2], dim=0)
        all_targets = torch.cat([targets_x, targets_x, targets_u, targets_u], dim=0)

        # 随机输出mini batch的序号，来mixup
        idx = torch.randperm(all_inputs.size(0))

        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]

        # 利用mix但是促使模型更偏向于label而不是UNlabel
        mixed_input = l * input_a + (1 - l) * input_b
        mixed_target = l * target_a + (1 - l) * target_b

        logits = net(mixed_input)
        # 输出被排列成两部分，input_x、Input_u
        logits_x = logits[:batch_size * 2]
        logits_u = logits[batch_size * 2:]

        # 利用公式(9)-(10)计算损失函数，其中lamb是所谓的warm up
        Lx, Lu, lamb = criterion(logits_x, mixed_target[:batch_size * 2],
                                 logits_u, mixed_target[batch_size * 2:],
                                 epoch + batch_idx / num_iter, warm_up)

        # regularization
        prior = torch.ones(args.num_class) / args.num_class
        prior = prior.cuda()
        pred_mean = torch.softmax(logits, dim=1).mean(0)
        # 一般来说会省略固定的prior部分，只取last term
        # lambR=1
        penalty = torch.sum(prior * torch.log(prior / pred_mean))

        # lamb是通过warm和current epoch比较得出的百分数，意味着随着epoch进行，Lu所占比重会逐渐增加
        # 前期需要保持标准CE损失，但是实际还有penalty
        # loss = Lx + lamb * Lu + penalty
        loss = Lx  + penalty + lamb * Lu
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_idx % 200 == 0:
            sys.stdout.write('\r')
            sys.stdout.write(
                '%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t Labeled loss: %.2f  Unlabeled loss: %.2f\n'
                % (args.dataset, args.r, args.noise_mode, epoch, args.num_epochs, batch_idx + 1, num_iter,
                   Lx.item(), Lu.item()))
            sys.stdout.flush()
    #         print('\r mse loss:%.4f\n' % mse_total, end='end', flush=True)
    # print('\r mse loss:%.4f\n' % mse_total, end='end', flush=True)

def mixup_criterion(pred, y_a, y_b, lam):
    c = F.log_softmax(pred, 1)
    return lam * F.cross_entropy(c, y_a) + (1 - lam) * F.cross_entropy(c, y_b)


soft_mix_warm = False

def warmup(epoch, net, optimizer, dataloader):
    net.train()
    num_iter = (len(dataloader.dataset) // dataloader.batch_size) + 1
    for batch_idx, (inputs, labels, path) in enumerate(dataloader):
        optimizer.zero_grad()
        l = np.random.beta(args.alpha, args.alpha)
        # 促使X'更加靠近labeled sample而不是无监督样本
        l = max(l, 1 - l)
        idx = torch.randperm(inputs.size(0))
        targets = torch.zeros(inputs.size(0), args.num_class).scatter_(1, labels.view(-1, 1), 1).cuda()
        targets = torch.clamp(targets, 1e-4, 1.)
        inputs, labels = inputs.cuda(), labels.cuda()
        if soft_mix_warm:
            input_a, input_b = inputs, inputs[idx]
            target_a, target_b = targets, targets[idx]
            labels_a, labels_b = labels, labels[idx]

            # 利用mix但是促使模型更偏向于label而不是UNlabel
            mixed_input = l * input_a + (1 - l) * input_b
            mixed_target = l * target_a + (1 - l) * target_b

            outputs = net(mixed_input)
            loss = mixup_criterion(outputs, labels_a, labels_b, l)
            L = loss
        else:
            outputs = net(inputs)
            loss = CEloss(outputs, labels)
            if args.noise_mode == 'asym':  # penalize confident prediction for asymmetric noise
                penalty = conf_penalty(outputs)
                L = loss + penalty
            elif args.noise_mode == 'sym':
                L = loss
        L.backward()
        optimizer.step()
        if batch_idx % 200 == 0:
            sys.stdout.write('\r')
            sys.stdout.write('%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t CE-loss: %.4f'
                             % (args.dataset, args.r, args.noise_mode, epoch, args.num_epochs, batch_idx + 1, num_iter,
                                loss.item()))
            sys.stdout.flush()


def test(epoch, net1, net2, best_acc, w_glob=None):
    if w_glob is None:
        net1.eval()
        net2.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                inputs, targets = inputs.cuda(), targets.cuda()
                outputs1 = net1(inputs)
                outputs2 = net2(inputs)
                outputs = outputs1 + outputs2
                _, predicted = torch.max(outputs, 1)

                total += targets.size(0)
                correct += predicted.eq(targets).cpu().sum().item()
        acc = 100. * correct / total
        if best_acc < acc:
            best_acc = acc
        print("\n| Ensemble network Test Epoch #%d\t Accuracy: %.2f, best_acc: %.2f%%\n" % (epoch, acc, best_acc))
        test_log.write('ensemble_Epoch:%d   Accuracy:%.2f, best_acc: %.2f\n' % (epoch, acc, best_acc))
        test_log.flush()
    else:
        net1_w_bak = net1.state_dict()
        net1.load_state_dict(w_glob)
        net1.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                inputs, targets = inputs.cuda(), targets.cuda()
                outputs1 = net1(inputs)
                _, predicted = torch.max(outputs1, 1)
                total += targets.size(0)
                correct += predicted.eq(targets).cpu().sum().item()
        acc = 100. * correct / total
        if best_acc < acc:
            best_acc = acc
        print("\n| Global network Test Epoch #%d\t Accuracy: %.2f, best_acc: %.2f%%\n" % (epoch, acc, best_acc))
        test_log.write('global_Epoch:%d   Accuracy:%.2f, best_acc: %.2f\n' % (epoch, acc, best_acc))
        test_log.flush()
        #   恢复权重
        net1.load_state_dict(net1_w_bak)
    return best_acc

feat_dim = 512  #是否可以加个全连接改成128
sim = torch.nn.CosineSimilarity(dim=1)

loss_func = torch.nn.CrossEntropyLoss(reduction='none')
def get_small_loss_samples(y_pred, y_true, forget_rate):
    loss = loss_func(y_pred, y_true)
    ind_sorted = np.argsort(loss.data.cpu()).cuda()
    loss_sorted = loss[ind_sorted]

    remember_rate = 1 - forget_rate
    num_remember = int(remember_rate * len(loss_sorted))

    ind_update = ind_sorted[:num_remember]

    return ind_update

def get_small_loss_by_loss_list(loss_list, forget_rate, eval_loader):
    remember_rate = 1 - forget_rate
    idx_list = []
    for i in range(10):
        class_idx = np.where(np.array(eval_loader.dataset.noise_label)[:] == i)[0]
        # class_idx = torch.from_numpy(class_idx).cuda()
        loss_per_class = loss_list[class_idx]   #取对应target的loss
        num_remember = int(remember_rate * len(loss_per_class))
        ind_sorted = np.argsort(loss_per_class.data.cpu())
        ind_update = ind_sorted[:num_remember].tolist()
        idx_list.append(ind_update)

    return idx_list

def eval_train(model, all_loss):
    model.eval()
    losses = torch.zeros(50000)
    f_G = torch.zeros(args.num_class, feat_dim).cuda()
    f_all = torch.zeros(50000, feat_dim).cuda()
    n_labels = torch.zeros(args.num_class, 1).cuda()
    y_k_tilde = torch.zeros(50000)
    mask = np.zeros(50000)
    with torch.no_grad():
        for batch_idx, (inputs, targets, index) in enumerate(eval_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs, feat = model(inputs, feat_out=True)
            loss = CE(outputs, targets)
            _, predicted = torch.max(outputs, 1)
            for b in range(inputs.size(0)):
                losses[index[b]] = loss[b]
                f_G[predicted[b]] += feat[b]
                n_labels[predicted[b]] += 1
            f_all[index] = feat
    assert torch.sum(n_labels) == 50000
    for i in range(len(n_labels)):
        if n_labels[i] == 0:
            n_labels[i] = 1
    f_G = torch.div(f_G, n_labels)
    f_G = F.normalize(f_G, dim=1)
    f_all = F.normalize(f_all, dim=1)
    temp = f_G.t()
    sim_all = torch.mm(f_all, temp)  # .cpu().numpy()
    y_k_tilde = torch.argmax(sim_all.cpu(), dim=1)
    with torch.no_grad():
        for batch_idx, (inputs, targets, index) in enumerate(eval_loader):
            for i in range(len(index)):
                if y_k_tilde[index[i]] == targets[i]:
                    mask[index[i]] = 1
    losses = (losses - losses.min()) / (losses.max() - losses.min())
    all_loss.append(losses)

    if args.r == 0.9:
        # average loss over last 5 epochs to improve convergence stability
        history = torch.stack(all_loss)
        input_loss = history[-5:].mean(0)
        input_loss = input_loss.reshape(-1, 1)
    else:
        input_loss = losses.reshape(-1, 1)

    # fit a two-component GMM to the loss
    # 参数如下：
    # n_components 聚类数量，max_iter 最大迭代次数，tol 阈值低于停止，reg_covar 协方差矩阵对角线上非负正则化参数，接近0即可
    gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
    gmm.fit(input_loss)
    prob = gmm.predict_proba(input_loss)
    prob = prob[:, gmm.means_.argmin()]
    return prob, all_loss, losses.numpy(), mask, f_G

def mix_data_lab(x, y, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()

    lam = max(lam, 1 - lam)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, index, lam


def linear_rampup(current, warm_up, rampup_length=16):
    # 线性warm_up，对sym噪声使用标准CE训练一段时间
    # 实际warm up epoch是warm_up+rampup_length
    current = np.clip((current - warm_up) / rampup_length, 0.0, 1.0)
    re_val = args.lambda_u * float(current)
    # print("   current warm up parameters:", current)
    # print("return parameters:", re_val)
    return re_val


class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch, warm_up):
        probs_u = torch.softmax(outputs_u, dim=1)

        # 利用mixup后的交叉熵，px输出*log(px_model)
        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        # 而UNlabel则是均方误差，p_u输出-pu_model
        Lu = torch.mean((probs_u - targets_u) ** 2)

        return Lx, Lu, linear_rampup(epoch, warm_up)


class NegEntropy(object):
    def __call__(self, outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log() * probs, dim=1))


def create_model():
    # 其实是pre-resnet18，使用的是pre-resnet block
    model = ResNet18(num_classes=args.num_class)
    model = model.cuda()
    return model

def plotHistogram(model_1_loss, model_2_loss, noise_index, clean_index, epoch, round, noise_rate):
    title = 'Epoch-' + str(epoch)+':'
    fig = plt.figure()
    plt.subplot(121)
    gmm = GaussianMixture(n_components=2, max_iter=20, tol=1e-2, random_state=0, reg_covar=5e-4)
    model_1_loss = np.reshape(model_1_loss, (-1, 1))
    gmm.fit(model_1_loss)  # fit the loss

    # plot resulting fit
    x_range = np.linspace(0, 1, 1000)
    pdf = np.exp(gmm.score_samples(x_range.reshape(-1, 1)))
    responsibilities = gmm.predict_proba(x_range.reshape(-1, 1))
    pdf_individual = responsibilities * pdf[:, np.newaxis]
    plt.hist(np.array(model_1_loss[noise_index]), density=True, bins=100, alpha=0.5,histtype='bar', color='red', label='Noisy subset')
    plt.hist(np.array(model_1_loss[clean_index]), density=True, bins=100, alpha=0.5,histtype='bar', color='blue', label='Clean subset')
    plt.plot(x_range, pdf, '-k', label='Mixture')
    plt.plot(x_range, pdf_individual, '--', label='Component')
    plt.legend(loc='upper right', prop={'size': 12})
    plt.xlabel('Normalized loss')
    plt.ylabel('Estimated pdf')
    plt.title(title+'Model_1')

    plt.subplot(122)
    gmm = GaussianMixture(n_components=2, max_iter=20, tol=1e-2, random_state=0, reg_covar=5e-4)
    model_2_loss = np.reshape(model_2_loss, (-1, 1))
    gmm.fit(model_2_loss)  # fit the loss

    # plot resulting fit
    x_range = np.linspace(0, 1, 1000)
    pdf = np.exp(gmm.score_samples(x_range.reshape(-1, 1)))
    responsibilities = gmm.predict_proba(x_range.reshape(-1, 1))
    pdf_individual = responsibilities * pdf[:, np.newaxis]
    plt.hist(np.array(model_2_loss[noise_index]), density=True, bins=100, alpha=0.5,histtype='bar', color='red', label='Noisy subset')
    plt.hist(np.array(model_2_loss[clean_index]), density=True, bins=100, alpha=0.5,histtype='bar', color='blue', label='Clean subset')
    plt.plot(x_range, pdf, '-k', label='Mixture')
    plt.plot(x_range, pdf_individual, '--', label='Component')
    plt.legend(loc='upper right', prop={'size': 12})
    plt.xlabel('Normalized loss')
    plt.ylabel('Estimated pdf')
    plt.title(title+'Model_2')

    print('\nlogging histogram...')
    title = 'cifar10_' + str(args.noise_mode) + '_moit_double_' + str(noise_rate)
    plt.savefig(os.path.join('./figure_his/', 'two_model_{}_{}_{}_{}.{}'.format(epoch, round, title, int(soft_mix_warm), ".tif")), dpi=300)
    # plt.show()
    plt.close()


def loss_dist_plot(loss, noisy_index, clean_index, epoch, rou=None, g_file=True, model_name='', loss2=None):
    """
    plot the loss distribution
    :param loss: the list contains the loss per sample
    :param noisy_index: contains the indices of real noisy label
    :param clean_index: contains the indices of real clean label
    :param filename: the generated pdf file name
    :param title: the figure title
    :param g_file: whether to generate the pdf figure file
    :return: None
    """
    if loss2 is None:
        filename = 'one_model_'+str(args.dataset)+'_'+str(args.noise_mode)+'_'+str(args.r)+'_epoch='+str(epoch)
        if rou is None:
            title = 'Epoch-'+str(epoch) + ': ' + str(args.dataset)+' '+str(args.r*100)+'%-'+str(args.noise_mode)
        else:
            title = 'Epoch-' + str(epoch) + ' ' +'Round-'+str(rou)+ ': ' + str(args.dataset) + ' ' + str(int(args.r * 100)) + '%-' + str(args.noise_mode)
        if type(loss) is not np.ndarray:
            loss= loss.numpy()
        sns.set(style='whitegrid')
        gmm = GaussianMixture(n_components=2, max_iter=20, tol=1e-2, random_state=0, reg_covar=5e-4)
        loss = np.reshape(loss, (-1, 1))
        gmm.fit(loss)  # fit the loss

        # plot resulting fit
        x_range = np.linspace(0, 1, 1000)
        pdf = np.exp(gmm.score_samples(x_range.reshape(-1, 1)))
        responsibilities = gmm.predict_proba(x_range.reshape(-1, 1))
        pdf_individual = responsibilities * pdf[:, np.newaxis]
        # sns.distplot(loss[noisy_index], color="red", rug=False,kde=False, label="incorrect",
        #              hist_kws={"color": "r", "alpha": 0.5})
        # sns.distplot(loss[clean_index], color="skyblue", rug=False,kde=False, label="correct",
        #              hist_kws={"color": "b", "alpha": 0.5})

        plt.hist(np.array(loss[noisy_index]), density=True, bins=100, histtype='bar', alpha=0.5, color='red',
                 label='Noisy subset')
        plt.hist(np.array(loss[clean_index]), density=True, bins=100, histtype='bar', alpha=0.5, color='blue',
                 label='Clean subset')
        plt.plot(x_range, pdf, '-k', label='Mixture')
        plt.plot(x_range, pdf_individual, '--', label='Component')
        # plt.plot(x_range, pdf_individual[:][1], '--', color='blue', label='Component 1')

        plt.title(title, fontsize=20)
        plt.xlabel('Normalized loss', fontsize=24)
        plt.ylabel('Estimated pdf', fontsize=24)

        plt.tick_params(labelsize=24)
        plt.legend(loc='upper right', prop={'size': 12})
        # plt.tight_layout()
        if g_file:
            plt.savefig('./figure_his/{0}.tif'.format(filename+model_name), bbox_inches='tight', dpi=300)
        #plt.show()
        plt.close()
    else:
        filename = 'noise_'+str(args.dataset) + '_' + str(args.noise_mode) + '_' + str(args.r) + '_epoch=' + str(epoch)
        if rou is None:
            title = 'Epoch-' + str(epoch) + ': ' + str(args.dataset) + ' ' + str(args.r * 100) + '%-' + str(
                args.noise_mode)
        else:
            title = 'Epoch-' + str(epoch) + ' ' + 'Round-' + str(rou) + ': ' + str(args.dataset) + ' ' + str(
                args.r * 100) + '%-' + str(args.noise_mode)
        if type(loss) is not np.ndarray:
            loss = loss.numpy()
        if type(loss2) is not np.ndarray:
            loss2 = loss2.numpy()
        fig = plt.figure()
        plt.subplot(121)
        sns.set(style='whitegrid')
        sns.distplot(loss[noisy_index], color="red", rug=False, kde=False, label="incorrect",
                     hist_kws={"color": "r", "alpha": 0.5})
        sns.distplot(loss[clean_index], color="skyblue", rug=False, kde=False, label="correct",
                     hist_kws={"color": "b", "alpha": 0.5})
        plt.title('Model_1', fontsize=32)
        plt.xlabel('Normalized loss', fontsize=32)
        plt.ylabel('Sample number', fontsize=32)
        plt.tick_params(labelsize=32)
        plt.legend(loc='upper right', prop={'size': 24})
        plt.subplot(122)
        sns.set(style='whitegrid')
        sns.distplot(loss2[noisy_index], color="red", rug=False, kde=False, label="incorrect",
                     hist_kws={"color": "r", "alpha": 0.5})
        sns.distplot(loss2[clean_index], color="skyblue", rug=False, kde=False, label="correct",
                     hist_kws={"color": "b", "alpha": 0.5})
        plt.title('Model_2', fontsize=32)
        plt.xlabel('Normalized loss', fontsize=32)
        plt.ylabel('Sample number', fontsize=32)
        plt.tick_params(labelsize=32)
        plt.legend(loc='upper right', prop={'size': 24})
        # plt.tight_layout()
        if g_file:
            plt.savefig('./figure_his/{0}.tif'.format(filename + model_name), bbox_inches='tight', dpi=300)
        # plt.show()
        plt.close()


def loss_dist_plot_real(loss, epoch, rou=None, g_file=True, model_name=''):
    """
    plot the loss distribution
    :param loss: the list contains the loss per sample
    :param noisy_index: contains the indices of real noisy label
    :param clean_index: contains the indices of real clean label
    :param filename: the generated pdf file name
    :param title: the figure title
    :param g_file: whether to generate the pdf figure file
    :return: None
    """
    filename = str(args.dataset) + '_' + str(args.noise_mode) + '_' + str(args.r) + '_epoch=' + str(epoch)
    if rou is None:
        title = 'Epoch-' + str(epoch) + ': ' + str(args.dataset) + ' ' + str(args.r * 100) + '%-' + str(args.noise_mode)
    else:
        title = 'Epoch-' + str(epoch) + ' ' + 'Round-' + str(rou) + ': ' + str(args.dataset) + ' ' + str(args.r * 100) + '%-' + str(args.noise_mode)

    if type(loss) is not np.ndarray:
        loss= loss.numpy()
    sns.set(style='whitegrid')

    gmm = GaussianMixture(n_components=2, max_iter=20, tol=1e-2, random_state=0, reg_covar=5e-4)
    loss = np.reshape(loss, (-1, 1))
    gmm.fit(loss)  # fit the loss

    # plot resulting fit
    x_range = np.linspace(0, 1, 1000)
    pdf = np.exp(gmm.score_samples(x_range.reshape(-1, 1)))
    responsibilities = gmm.predict_proba(x_range.reshape(-1, 1))
    pdf_individual = responsibilities * pdf[:, np.newaxis]

    plt.hist(loss, bins=60, density=True, histtype='bar', alpha=0.3)
    plt.plot(x_range, pdf, '-k', label='Mixture')
    plt.plot(x_range, pdf_individual, '--', label='Component')
    plt.legend()
    # plt.tight_layout()

    plt.title(title, fontsize=32)
    plt.xlabel('Normalized loss', fontsize=32)
    plt.ylabel('Estimated PDF', fontsize=32)
    plt.tick_params(labelsize=32)
    plt.legend(loc='upper right', prop={'size': 22})
    if g_file:
        plt.savefig('./figure_his/{0}.tif'.format(filename+model_name), bbox_inches='tight', dpi=300)
    #plt.show()
    plt.close()


def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
            # 只考虑iid noise的话，每个client训练样本数一样，所以不用做nk/n
        w_avg[k] = torch.div(w_avg[k], len(w))

    return w_avg


if os.path.exists('checkpoint') == False:
    os.mkdir('checkpoint')
    print("新建日志文件夹")
stats_log = open('./checkpoint/single_%s_%.1f_%s_%d' % (args.dataset, args.r, args.noise_mode,
                                                        int(soft_mix_warm)) + '_stats.txt', 'w')
test_log = open('./checkpoint/single_%s_%.1f_%s_%d' % (args.dataset, args.r, args.noise_mode,
                                                       int(soft_mix_warm)) + '_acc.txt', 'w')

warm_up = 10
dmix_epoch = 150
args.num_epochs = dmix_epoch + 150
# 第6页提及的warm up的epoch
if args.dataset == 'cifar10':
    warm_up = 10
    dmix_epoch = 150
    args.num_epochs = dmix_epoch + 50
elif args.dataset == 'cifar100':
    warm_up = 30
    dmix_epoch = 150
    args.num_epochs = dmix_epoch + 50

loader = dataloader.cifar_dataloader(args.dataset, r=args.r, noise_mode=args.noise_mode,
                                     batch_size=args.batch_size, num_workers=0,
                                     root_dir=args.data_path, log=stats_log,
                                     noise_file='%s/%.1f_%s.json' % (args.data_path, args.r, args.noise_mode))

print('| Building net')
net1 = create_model()
net2 = create_model()
cudnn.benchmark = True

criterion = SemiLoss()
optimizer1 = optim.SGD(net1.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
optimizer2 = optim.SGD(net2.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

CE = nn.CrossEntropyLoss(reduction='none')
CEloss = nn.CrossEntropyLoss()
if args.noise_mode == 'asym':
    # 本文第一个问题，对于非对称和对称需要不同措施，这很不适用
    # 其次本文在不同步骤中噪声数据处理措施很凌乱
    conf_penalty = NegEntropy()

all_loss = [[], []]  # save the history of losses from two networks

local_round = 5
first = True
balance_crit = 'median'
exp_path = './checkpoint/single_%s_%.1f_%s_double_m2_' % (args.dataset, args.r, args.noise_mode)
save_clean_idx = exp_path + "clean_idx.npy"
boot_loader = None
w_glob = None
if args.r == 0.9:
    args.p_threshold = 0.6
best_en_acc = 0.
best_gl_acc = 0.
resume_epoch = 0
if resume_epoch > 0:
    snapLast = exp_path + str(resume_epoch-1) + "_global_model.pth"
    global_state = torch.load(snapLast)
    # 先更新还是后跟新
    w_glob = global_state
    net1.load_state_dict(global_state)
    net2.load_state_dict(global_state)
for epoch in range(resume_epoch, args.num_epochs + 1):
    test_loader = loader.run('test')
    eval_loader = loader.run('eval_train')
    lr = args.lr
    if epoch >= dmix_epoch:
        lr /= 10
    for param_group in optimizer1.param_groups:
        param_group['lr'] = lr
    for param_group in optimizer2.param_groups:
        param_group['lr'] = lr

    noise_ind, clean_ind = eval_loader.dataset.if_noise()
    print(len(np.where(np.array(eval_loader.dataset.noise_label) != np.array(eval_loader.dataset.clean_label))[0])
          / len(eval_loader.dataset.clean_label))
    local_weights = []
    if epoch < warm_up:
        #   考虑warm up时是否需要merge
        warmup_trainloader = loader.run('warmup')
        print('Warmup Net1')
        warmup(epoch, net1, optimizer1, warmup_trainloader)
        print('\nWarmup Net2')
        warmup(epoch, net2, optimizer2, warmup_trainloader)
        if epoch == (warm_up-1):
            snapLast = exp_path+str(epoch) + "_1_model.pth"
            torch.save(net1.state_dict(), snapLast)
            snapLast = exp_path+str(epoch) + "_2_model.pth"
            torch.save(net1.state_dict(), snapLast)
            local_weights.append(net1.state_dict())
            local_weights.append(net2.state_dict())
            w_glob = FedAvg(local_weights)

    else:
        if epoch != warm_up:
            net1.load_state_dict(w_glob)
            net2.load_state_dict(w_glob)

        for rou in range(local_round):
            prob1, all_loss[0], loss1, mask1, f_G1 = eval_train(net1, all_loss[0])
            prob2, all_loss[1], loss2, mask2, f_G2 = eval_train(net2, all_loss[1])

            # 加载完global后第一次评估
            if rou == 0:
                # plotHistogram(np.array(loss1), np.array(loss2), noise_ind, clean_ind, epoch, rou, args.r)
                loss_dist_plot(loss1, noise_ind, clean_ind, epoch, model_name='model_1')
                # loss_dist_plot_real(loss1, epoch, model_name='model_1')
            if rou == local_round-1:
                plotHistogram(np.array(loss1), np.array(loss2), noise_ind, clean_ind, epoch, rou, args.r)

            # pred1 = (prob1 > args.p_threshold) & (mask1 != 0)
            # pred2 = (prob2 > args.p_threshold) & (mask2 != 0)
            pred1 = (prob1 > args.p_threshold)
            pred2 = (prob2 > args.p_threshold)

            non_zero_idx = pred1.nonzero()[0].tolist()
            aaa = len(non_zero_idx)
            if balance_crit == "max" or balance_crit == "min" or balance_crit == "median":
                num_clean_per_class = np.zeros(args.num_class)
                target_label = np.array(eval_loader.dataset.noise_label)[non_zero_idx]
                for i in range(args.num_class):
                    idx_class = np.where(target_label == i)[0]
                    num_clean_per_class[i] = len(idx_class)

                if balance_crit == "median":
                    num_samples2select_class = np.median(num_clean_per_class)

                for i in range(args.num_class):
                    idx_class = np.where(np.array(eval_loader.dataset.noise_label) == i)[0]
                    cur_num = num_clean_per_class[i]
                    idx_class2 = non_zero_idx
                    if num_samples2select_class > cur_num:
                        remian_idx = list(set(idx_class.tolist()) - set(idx_class2))
                        idx = list(range(len(remian_idx)))
                        random.shuffle(idx)
                        num_app = int(num_samples2select_class - cur_num)
                        idx = idx[:num_app]
                        for j in idx:
                            non_zero_idx.append(remian_idx[j])
            non_zero_idx = np.array(non_zero_idx).reshape(-1, )
            bbb = len(non_zero_idx)
            num_per_class2 = []
            for i in range(max(eval_loader.dataset.noise_label)):
                temp = np.where(np.array(eval_loader.dataset.noise_label)[non_zero_idx.tolist()] == i)[0]
                num_per_class2.append(len(temp))
            print('\npred1 appended num per class:', num_per_class2, aaa, bbb)
            idx_per_class = np.zeros_like(pred1).astype(bool)
            for i in non_zero_idx:
                idx_per_class[i] = True
            pred1 = idx_per_class
            non_aaa = pred1.nonzero()[0].tolist()
            assert len(non_aaa) == len(non_zero_idx)

            non_zero_idx2 = pred2.nonzero()[0].tolist()
            aaa = len(non_zero_idx2)
            if balance_crit == "max" or balance_crit == "min" or balance_crit == "median":
                num_clean_per_class = np.zeros(args.num_class)
                target_label = np.array(eval_loader.dataset.noise_label)[non_zero_idx2]
                for i in range(args.num_class):
                    idx_class = np.where(target_label == i)[0]
                    num_clean_per_class[i] = len(idx_class)

                if balance_crit == "median":
                    num_samples2select_class = np.median(num_clean_per_class)

                for i in range(args.num_class):
                    idx_class = np.where(np.array(eval_loader.dataset.noise_label) == i)[0]
                    cur_num = num_clean_per_class[i]
                    idx_class2 = non_zero_idx2
                    if num_samples2select_class > cur_num:
                        remian_idx = list(set(idx_class.tolist()) - set(idx_class2))
                        idx = list(range(len(remian_idx)))
                        random.shuffle(idx)
                        num_app = int(num_samples2select_class - cur_num)
                        idx = idx[:num_app]
                        for j in idx:
                            non_zero_idx2.append(remian_idx[j])
            non_zero_idx2 = np.array(non_zero_idx2).reshape(-1, )
            bbb = len(non_zero_idx2)
            num_per_class2 = []
            for i in range(max(eval_loader.dataset.noise_label)):
                temp = np.where(np.array(eval_loader.dataset.noise_label)[non_zero_idx2.tolist()] == i)[0]
                num_per_class2.append(len(temp))
            print('\npred2 appended num per class:', num_per_class2, aaa, bbb)
            idx_per_class2 = np.zeros_like(pred2).astype(bool)
            for i in non_zero_idx2:
                idx_per_class2[i] = True
            pred2 = idx_per_class2
            non_aaa = pred2.nonzero()[0].tolist()
            assert len(non_aaa) == len(non_zero_idx2)

            correct_num = len(pred1.nonzero()[0])
            eval_loader.dataset.if_noise(pred1)
            eval_loader.dataset.if_noise(pred2)

            print(f'round={rou}/{local_round}, dmix selection, Train Net1')
            # prob2就是先验概率wi,通过GMM拟合出来的，大于阈值就认为是clean，否则noisy
            labeled_trainloader, unlabeled_trainloader = loader.run('train', pred2, prob2)  # co-divide
            train(epoch, net1, net2, optimizer1, labeled_trainloader, unlabeled_trainloader)  # train net1

            print(f'\nround={rou}/{local_round}, dmix selection, Train Net2')
            labeled_trainloader, unlabeled_trainloader = loader.run('train', pred1, prob1)  # co-divide
            train(epoch, net2, net1, optimizer2, labeled_trainloader, unlabeled_trainloader)  # train net2

        local_weights.append(net1.state_dict())
        local_weights.append(net2.state_dict())
        w_glob = FedAvg(local_weights)
        if epoch % 5 == 0:
            snapLast = exp_path + str(epoch) + "_global_model.pth"
            torch.save(w_glob, snapLast)

    best_en_acc = test(epoch, net1, net2, best_en_acc)
    best_gl_acc= test(epoch, net1, net2, best_gl_acc, w_glob)



