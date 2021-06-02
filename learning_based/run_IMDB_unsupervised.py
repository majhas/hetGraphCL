from model_hetero import linear_eval
import torch
from sklearn.metrics import f1_score
import torch.nn as nn
from tools import evaluate_results_nc
import os
import numpy as np
from utils import load_data, EarlyStopping

def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight, gain=1.414)

def score(logits, labels):
    _, indices = torch.max(logits, dim=1)
    prediction = indices.long().cpu().numpy()
    labels = labels.cpu().numpy()

    accuracy = (prediction == labels).sum() / len(prediction)
    micro_f1 = f1_score(labels, prediction, average='micro')
    macro_f1 = f1_score(labels, prediction, average='macro')

    return accuracy, micro_f1, macro_f1

def evaluate(model, g, features, labels, mask, loss_func, model2):
    model.eval()
    model2.eval()
    with torch.no_grad():
        z = model.test_step(g, features)
        logits = model2(z)
    loss = loss_func(logits[mask], labels[mask])
    accuracy, micro_f1, macro_f1 = score(logits[mask], labels[mask])

    return loss, accuracy, micro_f1, macro_f1

def main(args):
    # If args['hetero'] is True, g would be a heterogeneous graph.
    # Otherwise, it will be a list of homogeneous graphs.
    g, features, labels, num_classes, train_idx, val_idx, test_idx, train_mask, \
    val_mask, test_mask = load_data(args['dataset'])

    best_micro_f1 = 0
    best_macro_f1 = 0

    if hasattr(torch, 'BoolTensor'):
        train_mask = train_mask.bool()
        val_mask = val_mask.bool()
        test_mask = test_mask.bool()

    features = features.to(args['device'])
    labels = labels.to(args['device'])
    train_mask = train_mask.to(args['device'])
    val_mask = val_mask.to(args['device'])
    test_mask = test_mask.to(args['device'])

    if args['hetero']:
        from model_hetero import HAN
        model = HAN(meta_paths=[['md', 'dm'], ['ma', 'am']],
                    in_size=features.shape[1],
                    hidden_size=args['hidden_units'],
                    out_size=num_classes,
                    num_heads=args['num_heads'],
                    dropout=args['dropout'])

        # model = nn.DataParallel(model)
        model.to(args['device'])
        g = g.to(args['device'])
    else:
        from model import HAN
        model = HAN(num_meta_paths=len(g),
                    in_size=features.shape[1],
                    hidden_size=args['hidden_units'],
                    out_size=num_classes,
                    num_heads=args['num_heads'],
                    dropout=args['dropout']).to(args['device'])
        g = [graph.to(args['device']) for graph in g]

    model_finetune = linear_eval(in_size=args['hidden_units'] * args['num_heads'][-1], out_size=num_classes)
    model_finetune = nn.DataParallel(model_finetune)
    model_finetune.to(args['device'])

    loss_fcn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'],
                                 weight_decay=args['weight_decay'])
    
    model_dir = './checkpoints'


    svm_macro_f1_lists = []
    svm_micro_f1_lists = []
    nmi_mean_list = []
    nmi_std_list = []
    ari_mean_list = []
    ari_std_list = []
    micro_f1_lists = []
    macro_f1_lists = []
    for i in range(10):
        best_micro_f1 = 0
        best_macro_f1 = 0
        best_nmi = 0
        best_ari = 0

        torch.save(model.state_dict(), os.path.join(model_dir, 'epoch-{}.pth'.format(0)))
        for epoch in range(args['num_epochs']):
            stopper = EarlyStopping(os.path.join(model_dir, 'model_epoch-{}.pth'.format(epoch)), patience=args['patience'])
            model.load_state_dict(torch.load(os.path.join(model_dir, 'epoch-{}.pth'.format(epoch))))
            model.train()
            model_finetune.train()
            model_finetune.apply(weight_reset)
            z = model(g, features, optimizer, epoch)
            torch.save(model.state_dict(), os.path.join(model_dir, 'epoch-{}.pth'.format(epoch+1)))
            stopper1 = EarlyStopping(os.path.join(model_dir, 'finetune_epoch-{}.pth'.format(epoch)), patience=args['patience'])

            params = list(model.parameters()) + list(model_finetune.parameters())
            finetune_opt = torch.optim.Adam(params, 0.001, 
                                        weight_decay=args['weight_decay'])
            # for i in range(200):
            #     z = model.test_step(g, features)
            #     logits = model_finetune(z)

            #     loss = loss_fcn(logits[train_mask], labels[train_mask])

            #     finetune_opt.zero_grad()

            #     loss.backward(retain_graph=True)
            #     finetune_opt.step()

            #     train_acc, train_micro_f1, train_macro_f1 = score(logits[train_mask], labels[train_mask])
            #     val_loss, val_acc, val_micro_f1, val_macro_f1 = evaluate(model, g, features, labels, val_mask, loss_fcn, model_finetune)
            #     early_stop = stopper1.step(val_loss.data.item(), val_acc, model_finetune)
            #     stopper.step(val_loss.data.item(), val_acc, model)

            #     # if (i + 1) % 10 == 0:
            #     #     print('Inner Epoch {:d} | Train Loss {:.4f} | Train Micro f1 {:.4f} | Train Macro f1 {:.4f} | '
            #     #         'Val Loss {:.4f} | Val Micro f1 {:.4f} | Val Macro f1 {:.4f}'.format(
            #     #         i + 1, loss.item(), train_micro_f1, train_macro_f1, val_loss.item(), val_micro_f1, val_macro_f1))

            #     if early_stop:
            #         break


            # stopper1.load_checkpoint(model_finetune)
            # stopper.load_checkpoint(model)
            with torch.no_grad():
                embeddings = model.test_step(g, features)

                svm_macro_f1_list, svm_micro_f1_list, nmi_mean, nmi_std, ari_mean, ari_std = evaluate_results_nc(
                    embeddings.cpu().numpy(), labels.cpu().numpy(), num_classes=num_classes)
            
            
            # test_loss, test_acc, test_micro_f1, test_macro_f1 = evaluate(model, g, features, labels, test_mask, loss_fcn, model_finetune)
            # print('Test loss {:.4f} | Test Micro f1 {:.4f} | Test Macro f1 {:.4f}'.format(
            #     test_loss.item(), test_micro_f1, test_macro_f1))



            best_micro_f1 = max(best_micro_f1, svm_micro_f1_list[3][0])
            best_macro_f1 = max(best_macro_f1, svm_macro_f1_list[3][0])
            best_nmi = max(best_nmi, nmi_mean)
            best_ari = max(best_ari, ari_mean)
            
        print('Best Test Micro f1 {:.4f} | Best Test Macro f1 {:.4f}'.format(best_micro_f1, best_macro_f1))
        micro_f1_lists.append(best_micro_f1)
        macro_f1_lists.append(best_macro_f1)
        nmi_mean_list.append(best_nmi)
        ari_mean_list.append(best_ari)        
    print('Test Micro f1 AVG {:.4f} | Test Micro f1 STD {:.4f} | Test Macro f1 AVG {:.4f} | Test Macro f1 STD {:.4f} |\
        NMI AVG {:.4f} | NMI STD {:.4f} | ARI AVG {:.4f} | ARI STD {:.4f}'.format(
        np.mean(micro_f1_lists), np.std(micro_f1_lists), np.mean(macro_f1_lists), np.std(macro_f1_lists),
        np.mean(nmi_mean_list), np.std(nmi_mean_list), np.mean(ari_mean_list), np.std(ari_mean_list)
    ))

if __name__ == '__main__':
    import argparse

    from utils import setup

    parser = argparse.ArgumentParser('HAN')
    parser.add_argument('-dataset', type=str, default='IMDB')
    parser.add_argument('-s', '--seed', type=int, default=1,
                        help='Random seed')
    parser.add_argument('-ld', '--log-dir', type=str, default='results',
                        help='Dir for saving training results')
    parser.add_argument('--hetero', action='store_true', default=True,
                        help='Use metapath coalescing with DGL\'s own dataset')
    args = parser.parse_args().__dict__

    args = setup(args)

    main(args)