import numpy as np
import os.path as osp
from collections import OrderedDict, defaultdict
import torch
from sklearn.metrics import f1_score, confusion_matrix
from scipy.stats import pearsonr

from .build import EVALUATOR_REGISTRY
import os

class EvaluatorBase:
    """Base evaluator."""

    def __init__(self, cfg):
        self.cfg = cfg

    def reset(self):
        raise NotImplementedError

    def process(self, mo, gt):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError


@EVALUATOR_REGISTRY.register()
class Classification(EvaluatorBase):
    """Evaluator for classification."""

    def __init__(self, cfg, lab2cname=None, **kwargs):
        super().__init__(cfg)
        self._lab2cname = lab2cname
        self._correct = 0
        self._total = 0
        self._per_class_res = None
        self._per_class_res_thres = None
        self._y_true = []
        self._y_pred = []
        if cfg.TEST.PER_CLASS_RESULT:
            assert lab2cname is not None
            self._per_class_res = defaultdict(list)
            self._per_class_res_thres = defaultdict(list)
        self.conf_thre = cfg.TRAINER.FIXMATCH.CONF_THRE
        self.directory = self.cfg.OUTPUT_DIR
        print(self.directory)

    def reset(self):
        self._correct = 0
        self._total = 0
        self._y_true = []
        self._y_pred = []
        if self._per_class_res is not None:
            self._per_class_res = defaultdict(list)
            self._per_class_res_thres = defaultdict(list) # process for collect the pseudo-label prediction information of training set in ssl setting

    def process(self, mo, gt, len_dom=0):
        # mo (torch.Tensor): model output [batch, num_classes]
        # gt (torch.LongTensor): ground truth [batch]
        
        mo = torch.softmax(mo.detach()/1, dim=-1)
        max_prob, pred = mo.max(1)

        if len_dom:
            # print('only care about the classification accuracy, without considering the domain accuracy!')
            device = pred.device
            pred = [out % len_dom for out in pred ]
            pred = torch.tensor(pred).to(device)
            gt = [out % len_dom for out in gt ]
            gt = torch.tensor(gt).to(device)
            # 同时对predictions和ground-truth都映射到相同类即可

        matches = pred.eq(gt).float()
        if self._per_class_res_thres is not None:
            mask = max_prob.ge(self.conf_thre).float() #pseudo-label utilized
            matches_thres = matches * mask #pseudo-label utilized right
            

        self._correct += int(matches.sum().item())
        self._total += gt.shape[0]

        self._y_true.extend(gt.data.cpu().numpy().tolist())
        self._y_pred.extend(pred.data.cpu().numpy().tolist())
        if self._per_class_res is not None:
            for i, label in enumerate(gt):
                label = label.item()
                matches_i = int(matches[i].item())
                self._per_class_res[label].append(matches_i)
                if self._per_class_res_thres is not None:
                    matches_thres_i = int(matches_thres[i].item())
                    mask_i = int(mask[i].item())
                    self._per_class_res_thres[label].append((mask_i, matches_thres_i))

    
    
    def evaluate(self, domain=''):
        results = OrderedDict()
        acc = 100.0 * self._correct / self._total
        err = 100.0 - acc
        macro_f1 = 100.0 * f1_score(
            self._y_true,
            self._y_pred,
            average="macro",
            labels=np.unique(self._y_true)
        )

        # The first value will be returned by trainer.test()
        results["accuracy"] = acc
        results["error_rate"] = err
        results["macro_f1"] = macro_f1

        print(
            "=> result\n"
            f"* total: {self._total:,}\n"
            f"* correct: {self._correct:,}\n"
            f"* accuracy: {acc:.1f}%\n"
            f"* error: {err:.1f}%\n"
            f"* macro_f1: {macro_f1:.1f}%"
        )

        if self._per_class_res is not None:
            labels = list(self._per_class_res.keys())
            labels.sort()

            print("=> per-class result")
            accs = []
            accs_thres = []
            statics = defaultdict(list)

            for label in labels:
                classname = self._lab2cname[label]
                res = self._per_class_res[label]
                correct = sum(res)
                total = len(res)
                acc = 100.0 * correct / total
                accs.append(acc)
                # statics['label'].append(str(label)+':' + classname)
                statics['label'].append(str(label))
                statics['classname'].append(classname)
                statics['total'].append(total)
                statics['acc'].append(acc)
                statics['correct'].append(correct)

                if self._per_class_res_thres is not None:
                    res_thres = self._per_class_res_thres[label]
                    res_thres = list(map(list, zip(*res_thres)))
                    correct_thres = sum(res_thres[1])
                    total_thres = sum(res_thres[0])
                    # print(res_thres)
                    # print(correct_thres)
                    # print(total_thres)
                    

                    if total_thres:
                        acc_thres = 100.0 * correct_thres / total_thres
                    else:
                        acc_thres = 0.0
                    accs_thres.append(acc_thres)
                                    
                    statics['total_thres'].append(total_thres)
                    statics['acc_thres'].append(acc_thres)
                    statics['correct_thres'].append(correct_thres)
                    print(
                    f"* class: {label} ({classname})\t"
                    f"total: {total:,}\t"
                    f"correct: {correct:,}\t"
                    f"acc: {acc:.1f}%\t"
                    f"total_thres: {total_thres:,}\t"
                    f"correct_thres: {correct_thres:,}\t"
                    f"acc_thres: {acc_thres:.1f}%"
                )
                else:
                    print(
                        f"* class: {label} ({classname})\t"
                        f"total: {total:,}\t"
                        f"correct: {correct:,}\t"
                        f"acc: {acc:.1f}%"
                    )

            mean_acc = np.mean(accs)
            print(f"* average: {mean_acc:.1f}%")
            results["perclass_accuracy"] = mean_acc

            if self._per_class_res_thres is not None:
                mean_acc_thres = np.mean(accs_thres)
                print(f"* average thres: {mean_acc_thres:.1f}%")
                results["perclass_accuracy_thres"] = mean_acc_thres
                # correlation_tc = pearsonr(statics['total_thres'], statics['correct_thres'])
                correlation_ta_thres = pearsonr(statics['total_thres'], statics['acc_thres'])
                # correlation_ca = pearsonr(statics['correct_thres'], statics['acc_thres'])
                # print(f"* correlation_tc thres: {correlation_tc[0]}%")
                correlation_ta = pearsonr(statics['total'], statics['acc'])
                print(f"* correlation_ta thres: {correlation_ta[0]}%")
                # print(f"* correlation_ca thres: {correlation_ca[0]}%")
                statics['correlation_ta'].extend([correlation_ta_thres, correlation_ta])

            if domain:
                self.draw(statics, domain)
            else:
                self.write_results(statics)

        if self.cfg.TEST.COMPUTE_CMAT:
            cmat = confusion_matrix(
                self._y_true, self._y_pred, normalize="true"
            )
            save_path = osp.join(self.cfg.OUTPUT_DIR, "cmat.pt")
            torch.save(cmat, save_path)
            print(f"Confusion matrix is saved to {save_path}")

        return results

    def write_results(self, statics):
        save_path = osp.join(self.cfg.OUTPUT_DIR, "test_statics.txt")
        classname = statics['classname']
        total = statics['total']
        acc = statics['acc']
        correct = statics['correct']
        

        sorted_id = sorted(range(len(acc)), key=lambda k: acc[k], reverse=True)  # 元素索引序列：降序
        classname = [classname[i] for i in sorted_id]
        total = [total[i] for i in sorted_id]
        acc = [acc[i] for i in sorted_id]
        correct = [correct[i] for i in sorted_id]
        tplt = "{:<30}\t{:<10}\t{:<10}\t{:.2f}%"

        with open(save_path, "w") as f:
            f.write('Category                        total       correct     accuracy')
            # 换行
            f.write('\n')
            for name, t, c, a in zip(classname, total, correct, acc):
                f.write(tplt.format(name, t, c, a, chr(255)))
                f.write('\n')
        print(f"Results are written to {save_path}")

    def draw(self, statics, domain):
        from matplotlib import pyplot as plt
        x = statics['label']

        total = statics['total_thres']
        correct = statics['correct_thres']
        acc = statics['acc_thres']

        # sort
        sorted_id = sorted(range(len(total)), key=lambda k:total[k], reverse=True)
        x = np.array(x)[sorted_id]
        total = np.array(total)[sorted_id]
        correct = np.array(correct)[sorted_id]
        acc = np.array(acc)[sorted_id]       

        mean_acc = np.mean(acc)
        mean_total = np.mean(total)
        mean_correct = np.mean(correct)

        fig, ax = plt.subplots(2,3)
        # fig, ax = plt.subplots(2,2)
        fig.suptitle(f'Statistical information on pseudo-labels: {domain}')
        ax[0][0].set_title(f'total_thres & correct_thres', fontsize=6)
        ax[0][0].bar(x, total, width=1, label=f'total:{mean_total:.2f}')
        ax[0][0].bar(x, correct, width=1, label=f'correct:{mean_correct:.2f}')
        ax[0][0].set_xticks([])
        ax[0][0].legend()
        
        ax[1][0].set_title(f'acc_thres', fontsize=6)
        ax[1][0].bar(x, acc, width=1, label=f'acc:{mean_acc:.2f}')
        ax[1][0].set_xticks([])
        ax[1][0].legend()

        sorted_id = sorted(range(len(correct)), key=lambda k:correct[k], reverse=True)
        x = np.array(x)[sorted_id]
        total = np.array(total)[sorted_id]
        correct = np.array(correct)[sorted_id]
        acc = np.array(acc)[sorted_id]

        ax[0][1].set_title(f'total_thres & correct_thres', fontsize=6)
        ax[0][1].bar(x, total, width=1, label=f'total:{mean_total:.2f}')
        ax[0][1].bar(x, correct, width=1, label=f'correct:{mean_correct:.2f}')
        ax[0][1].set_xticks([])
        ax[0][1].legend()
        
        ax[1][1].set_title(f'acc_thres', fontsize=6)
        ax[1][1].bar(x, acc, width=1, label=f'acc:{mean_acc:.2f}')
        ax[1][1].set_xticks([])
        ax[1][1].legend()


        # original statics without thresholding
        total = statics['total']
        correct = statics['correct']
        acc = statics['acc']
        sorted_id = sorted(range(len(correct)), key=lambda k:correct[k], reverse=True)
        x = np.array(x)[sorted_id]
        total = np.array(total)[sorted_id]
        correct = np.array(correct)[sorted_id]
        acc = np.array(acc)[sorted_id]
        mean_acc = np.mean(acc)
        mean_total = np.mean(total)
        mean_correct = np.mean(correct)

        ax[0][2].set_title(f'total& correct', fontsize=6)
        ax[0][2].bar(x, total, width=1, label=f'total:{mean_total:.2f}')
        ax[0][2].bar(x, correct, width=1, label=f'correct:{mean_correct:.2f}')
        ax[0][2].set_xticks([])
        ax[0][2].legend()
        
        ax[1][2].set_title(f'acc', fontsize=6)
        ax[1][2].bar(x, acc, width=1, label=f'acc:{mean_acc:.2f}')
        ax[1][2].set_xticks([])
        ax[1][2].legend()


        if not osp.exists(osp.join(self.directory, 'figures')):
            os.makedirs(osp.join(self.directory, 'figures'))
        plt.savefig(osp.join(self.directory, 'figures', domain+'_statistics.png'))
        plt.close()

    def cal_corr(self, statics, domain):
        pass

