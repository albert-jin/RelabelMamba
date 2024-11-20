import argparse
import genericpath
import time

import torch.optim.lr_scheduler
import torchvision.transforms as transforms

import models
import wandb
from torch.optim import SGD
from torch.utils.data import Subset
from tqdm import tqdm

from datasets.dataloader_cifar import cifar_dataset
from models.dinov2 import DinoVisionTransformerClassifier
# from models.mae_Jig import create_model
from timm.models import create_model

from models.preresnet import PreResNet18
from utils import *
from vim.models_mamba import vim_small_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2, \
    segm_init_weights, vim_small_patch16_stride8_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2

parser = argparse.ArgumentParser('Train with synthetic cifar noisy dataset')
parser.add_argument('--dataset_path', default='/data/data/academic', help='dataset path')
parser.add_argument('--noisy_dataset_path', default='/data/data/academic', help='open-set noise dataset path')
parser.add_argument('--dataset', default='cifar100', help='dataset name') # cifar100
parser.add_argument('--noisy_dataset', default='cifar10', help='open-set noise dataset name')

# dataset settings
parser.add_argument('--noise_mode', default='sym', type=str, help='artifical noise mode (default: symmetric)')  # asym  sym
parser.add_argument('--noise_ratio', default=0.8, type=float, help='artifical noise ratio (default: 0.5)')  # 0.5
parser.add_argument('--open_ratio', default=0.0, type=float, help='artifical noise ratio (default: 0.0)')

# model settings
parser.add_argument('--theta_s', default=1.0, type=float, help='threshold for selecting samples (default: 1)')
parser.add_argument('--theta_r', default=0.9, type=float, help='threshold for relabelling samples (default: 0.9)')
parser.add_argument('--lambda_fc', default=1.0, type=float, help='weight of feature consistency loss (default: 1.0)')
parser.add_argument('--k', default=200, type=int, help='neighbors for knn sample selection (default: 200)')

# train settings
parser.add_argument('--model', default='PreResNet18', help=f'model architecture (default: PreResNet18)')
parser.add_argument('--epochs', default=50, type=int, metavar='N', help='number of total epochs to run (default: 300)')            # 300
parser.add_argument('--batch_size', default=22, type=int, help='mini-batch size (default: 128)')  # 128
parser.add_argument('--lr', default=0.0002, type=float, help='initial learning rate (default: 0.02)')  # 0.02
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum of SGD solver (default: 0.9)')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay (default: 5e-4)')
parser.add_argument('--seed', default=3047, type=int, help='seed for initializing training. (default: 3047)')
parser.add_argument('--gpuid', default='0', type=str, help='Selected GPU (default: "0")')
parser.add_argument('--entity', type=str, help='Wandb user entity')  # , default='21e471e150b8b942b9032ee23b4ebe31c2eb85ee'
parser.add_argument('--datadb_project', default='ssr', help='Wandb user project')

parser.add_argument('--run_path', type=str, help='run path containing all results')


def adaptive_mixup_coefficient(epoch, max_epochs):
    alpha = 4 * (1 - epoch / max_epochs)
    beta = 4 * (1 - epoch / max_epochs)
    return np.random.beta(alpha, beta)

def puzzle_mix(input_a, input_b, n_splits=4):
    split_size = input_a.size(2) // n_splits
    patches_a = input_a.unfold(2, split_size, split_size).unfold(3, split_size, split_size).permute(0, 2, 3, 1, 4, 5)
    patches_b = input_b.unfold(2, split_size, split_size).unfold(3, split_size, split_size).permute(0, 2, 3, 1, 4, 5)

    # Randomly mix patches from a and b
    mix_flag = torch.rand(n_splits, n_splits) > 0.5
    for i in range(n_splits):
        for j in range(n_splits):
            if mix_flag[i, j]:
                patches_a[:, i, j] = patches_b[:, i, j]

    # Reconstruct images from mixed patches
    mixed_input = patches_a.permute(0, 3, 1, 4, 2, 5).reshape(input_a.size())
    return mixed_input




def train(labeled_trainloader, modified_label, all_trainloader, encoder, classifier, proj_head, pred_head, optimizer,
          epoch, args):
    # encoder.train()
    encoder.eval()
    classifier.train()
    proj_head.train()
    pred_head.train()
    xlosses = AverageMeter('xloss')
    labeled_train_iter = iter(labeled_trainloader)
    all_bar = tqdm(all_trainloader)

    for [inputs_x1, inputs_x2], labels_x, _, index in labeled_train_iter:
        # cross-entropy training with Adaptive Mixup
        batch_size = inputs_x1.size(0)
        inputs_x1, inputs_x2 = inputs_x1.cuda(), inputs_x2.cuda()
        labels_x = modified_label[index]
        targets_x = torch.zeros(batch_size, args.num_classes, device=inputs_x1.device).scatter_(1, labels_x.view(-1, 1), 1)
        l = adaptive_mixup_coefficient(epoch, args.epochs)  # Get mixup coefficient based on current epoch
        l = max(l, 1 - l)
        all_inputs_x = torch.cat([inputs_x1, inputs_x2], dim=0)
        all_targets_x = torch.cat([targets_x, targets_x], dim=0)
        idx = torch.randperm(all_inputs_x.size()[0])
        input_a, input_b = all_inputs_x, all_inputs_x[idx]
        target_a, target_b = all_targets_x, all_targets_x[idx]

        mixed_input = puzzle_mix(input_a, input_b)

        mixed_target = l * target_a + (1 - l) * target_b

        logits = classifier(encoder(mixed_input))
        Lce = -torch.mean(torch.sum(F.log_softmax(logits, dim=1) * mixed_target, dim=1))


        loss = Lce
        xlosses.update(Lce.item())
        all_bar.set_description(
            f'Train epoch {epoch} LR:{optimizer.param_groups[0]["lr"]} Labeled loss: {xlosses.avg:.4f}')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def test(testloader, encoder, classifier, epoch):
    encoder.eval()
    classifier.eval()
    accuracy = AverageMeter('accuracy')
    data_bar = tqdm(testloader)
    with torch.no_grad():
        for i, (data, label, _) in enumerate(data_bar):
            data, label = data.cuda(), label.cuda()
            feat = encoder(data)
            res = classifier(feat)
            # res = res.squeeze(-1)

            pred = torch.argmax(res, dim=1)
            acc = torch.sum(pred == label) / float(data.size(0))
            accuracy.update(acc.item(), data.size(0))
            data_bar.set_description(f'Test epoch {epoch}: Accuracy#{accuracy.avg:.4f}')
    # logger.log({'acc': accuracy.avg})
    return accuracy.avg

def evaluate(dataloader, encoder, classifier, args, noisy_label, clean_label, i, stat_logs):
    encoder.eval()
    classifier.eval()
    feature_bank_cpu = []
    prediction = []

    feature_arrbank = []
    ################################### feature extraction ###################################
    with torch.no_grad():

        for (data, target, _, index) in tqdm(dataloader, desc='Feature extracting'):
            data = data.cuda()
            feature = encoder(data, is_evaluate = True)
            # feature_cpu = feature.to("cpu")
            # feature_bank_cpu.append(feature_cpu)

            feature_arr = feature.cpu().numpy()
            feature_arrbank.append(feature_arr)
            # values = feature_array.tolist()

            feature = feature.unsqueeze(-1)
            res = classifier(feature)

            prediction.append(res)


        
        concatenated_array = np.concatenate(feature_arrbank, axis=0)

        feature_bank = torch.from_numpy(concatenated_array)

        feature_bank = F.normalize(feature_bank, dim=1).to(torch.device("cuda:0"))

        ################################### sample relabelling ###################################
        prediction_cls = torch.softmax(torch.cat(prediction, dim=0), dim=1)
        his_score, his_label = prediction_cls.max(1)
        print(f'Prediction track: mean: {his_score.mean()} max: {his_score.max()} min: {his_score.min()}')

        entropy = -torch.sum(prediction_cls * torch.log(prediction_cls + 1e-5), dim=1)
        top_element = int(entropy.numel() * (i + 1) / args.epochs)
        top_values, conf_id = torch.topk(-entropy, top_element)
        modified_label = torch.clone(noisy_label).detach()
        modified_label[conf_id] = his_label[conf_id]

        ################################### sample selection ###################################
        prediction_knn = weighted_knn(feature_bank, feature_bank, modified_label, args.num_classes, args.k, 10)
        vote_y = torch.gather(prediction_knn, 1, modified_label.view(-1, 1)).squeeze()
        vote_max = prediction_knn.max(dim=1)[0]
        right_score = vote_y / vote_max
        clean_id = torch.where(right_score >= args.theta_s)[0]

        total = len(prediction_cls)
        mask = torch.ones(total, dtype = torch.bool, device = conf_id.device)
        mask[conf_id] = False
        noisy_id = torch.nonzero(mask, as_tuple = True)[0]


        ################################### SSR monitor ###################################
        TP = torch.sum(modified_label[clean_id] == clean_label[clean_id])
        FP = torch.sum(modified_label[clean_id] != clean_label[clean_id])
        TN = torch.sum(modified_label[noisy_id] != clean_label[noisy_id])
        FN = torch.sum(modified_label[noisy_id] == clean_label[noisy_id])
        print(f'Epoch [{i}/{args.epochs}] selection: theta_s:{args.theta_s} TP: {TP} FP:{FP} TN:{TN} FN:{FN}')

        with open("modified_calculate.txt", 'a') as modified_file:

            for idx in range(args.num_classes):

                clean_idx = clean_label == idx
                class_count = torch.sum(clean_idx).item()
                class_right = torch.sum((clean_label == modified_label) & clean_idx).item()
                # if class_count > 0:
                modified_file.write("Epoch " + str(i) +  " 类 "+ str(idx) + "修改modified_label准确率为：" + str(float(class_right) / class_count) + "\n")
            modified_file.write("\n")

        with open("epoch_tp.txt", 'a') as epoch_file:
            epoch_file.write(f'Epoch {i} TP: {TP} FP:{FP} TN:{TN} FN:{FN}' + "\n")

        with open("clean_select.txt", 'a') as clean_file:

            predicted_clean_label = clean_label[clean_id]
            modified_label_control = modified_label[clean_id]
            clean_file.write("Epoch " + str(i) + " 为correct有： " + str(clean_id.size(0)) + " 其中正确的有： " + str(torch.sum(modified_label_control == predicted_clean_label).item()) + "\n")
            for idx in range(args.num_classes):
                clean_idx = predicted_clean_label == idx
                class_count = torch.sum(clean_idx).item()
                class_right = torch.sum((predicted_clean_label == modified_label_control) & clean_idx).item()
                if class_count > 0:
                    clean_file.write("Epoch " + str(i) +  " 类 "+ str(idx) + "确定为clean的准确率为：" + str(float(class_right) / class_count) + "\n")

        correct = torch.sum(modified_label[conf_id] == clean_label[conf_id])
        orginal = torch.sum(noisy_label[conf_id] == clean_label[conf_id])
        all = len(conf_id)
        # logger.log({'correct': correct, 'original': orginal, 'total': all})
        print(f'Epoch [{i}/{args.epochs}] relabelling:  correct: {correct} original: {orginal} total: {all}')

        stat_logs.write(f'Epoch [{i}/{args.epochs}] selection: theta_s:{args.theta_s} TP: {TP} FP:{FP} TN:{TN} FN:{FN}\n')
        stat_logs.write(f'Epoch [{i}/{args.epochs}] relabelling:  correct: {correct} original: {orginal} total: {all}\n')
        stat_logs.flush()
    return clean_id, noisy_id, modified_label


def main():
    args = parser.parse_args()
    seed_everything(args.seed)
    if args.run_path is None:
        args.run_path = f'Dataset({args.dataset}_{args.noise_ratio}_{args.open_ratio}_{args.noise_mode})_Model({args.theta_r}_{args.theta_s})'

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid
    global logger
    logger = wandb.init(project=args.datadb_project, entity=args.entity, name=args.run_path, group=args.dataset)
    logger.config.update(args)

    # generate noisy dataset with our transformation
    if not os.path.isdir(f'{args.dataset}'):
        os.mkdir(f'{args.dataset}')
    if not os.path.isdir(f'{args.dataset}/{args.run_path}'):
        os.mkdir(f'{args.dataset}/{args.run_path}')

    ############################# Dataset initialization ##############################################
    if args.dataset == 'cifar10':
        args.num_classes = 10
        # args.image_size = 32  224
        args.image_size = 224
        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    elif args.dataset == 'cifar100':
        args.num_classes = 100
        # args.image_size = 32
        args.image_size = 224
        normalize = transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276))
    else:
        raise ValueError(f'args.dataset should be cifar10 or cifar100, rather than {args.dataset}!')

    # dinov2 input_size
    image_dimension = 224  # 256
    target_size = (image_dimension, image_dimension)

    # data loading
    weak_transform = transforms.Compose([
        ResizeAndPad(target_size, 14),
        # transforms.RandomResizedCrop(224),
        # transforms.RandomCrop(image_dimension, padding=4),
        # transforms.RandomRotation(360),
        transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        normalize])
    none_transform = transforms.Compose([
        ResizeAndPad(target_size, 14),
        # transforms.RandomResizedCrop(224),
        transforms.ToTensor(), normalize])
    strong_transform = transforms.Compose([
        ResizeAndPad(target_size, 14),
        # transforms.RandomResizedCrop(224),
        # transforms.RandomCrop(image_dimension, padding=4),
        transforms.RandomRotation(360),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        # CIFAR10Policy(),
        transforms.ToTensor(),
        normalize])

    # generate train dataset with only filtered clean subset
    train_data = cifar_dataset(dataset=args.dataset, root_dir=args.dataset_path,
                               noise_data_dir=args.noisy_dataset_path, noisy_dataset=args.noisy_dataset,
                               transform=KCropsTransform(strong_transform, 2), open_ratio=args.open_ratio,
                               dataset_mode='train', noise_ratio=args.noise_ratio, noise_mode=args.noise_mode,
                               noise_file=f'{args.dataset}_{args.noise_ratio}_{args.open_ratio}_{args.noise_mode}_noise.json')
    eval_data = cifar_dataset(dataset=args.dataset, root_dir=args.dataset_path, transform=weak_transform,
                              noise_data_dir=args.noisy_dataset_path, noisy_dataset=args.noisy_dataset,
                              dataset_mode='train', noise_ratio=args.noise_ratio, noise_mode=args.noise_mode,
                              open_ratio=args.open_ratio,
                              noise_file=f'{args.dataset}_{args.noise_ratio}_{args.open_ratio}_{args.noise_mode}_noise.json')
    test_data = cifar_dataset(dataset=args.dataset, root_dir=args.dataset_path, transform=none_transform,
                              noise_data_dir=args.noisy_dataset_path, noisy_dataset=args.noisy_dataset,
                              dataset_mode='test')
    all_data = cifar_dataset(dataset=args.dataset, root_dir=args.dataset_path,
                                   noise_data_dir=args.noisy_dataset_path, noisy_dataset=args.noisy_dataset,
                                   transform=MixTransform(strong_transform=strong_transform, weak_transform=weak_transform, K=1),
                                   open_ratio=args.open_ratio,
                                   dataset_mode='train', noise_ratio=args.noise_ratio, noise_mode=args.noise_mode,
                                   noise_file=f'{args.dataset}_{args.noise_ratio}_{args.open_ratio}_{args.noise_mode}_noise.json')

    # extract noisy labels and clean labels for performance monitoring
    noisy_label = torch.tensor(eval_data.cifar_label).cuda()
    clean_label = torch.tensor(eval_data.clean_label).cuda()

    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    eval_loader = torch.utils.data.DataLoader(eval_data, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    all_loader = torch.utils.data.DataLoader(all_data, batch_size=args.batch_size, num_workers=4, shuffle=True, pin_memory=True, drop_last=True)

    ################################ Model initialization ###########################################
    
    encoder = vim_small_patch16_stride8_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2(pretrained=False,
        num_classes=args.num_classes,
        drop_rate=0.0,
        drop_path_rate=0.1,
        drop_block_rate=None,
        img_size=args.image_size
    )

    checkpoint = torch.load("vim/pretrained/vim_s_midclstok_ft_81p6acc.pth", map_location='cpu') # vim_s_midclstok_80p5acc

    checkpoint_model = checkpoint['model']

    # interpolate position embedding
    pos_embed_checkpoint = checkpoint_model['pos_embed']
    embedding_size = pos_embed_checkpoint.shape[-1]
    num_patches = encoder.patch_embed.num_patches
    num_extra_tokens = encoder.pos_embed.shape[-2] - num_patches
    # height (== width) for the checkpoint position embedding
    orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
    # height (== width) for the new position embedding
    new_size = int(num_patches ** 0.5)
    # class_token and dist_token are kept unchanged
    extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
    # only the position tokens are interpolated
    pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
    pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
    pos_tokens = torch.nn.functional.interpolate(
        pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
    pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
    new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
    checkpoint_model['pos_embed'] = new_pos_embed

    encoder.load_state_dict(checkpoint_model, strict=False)



    embed_dim = 384     # 不同的模型大小不一样   # 384
   
    classifier = torch.nn.Sequential(# torch.nn.Linear(embed_dim, 256), torch.nn.BatchNorm1d(256), torch.nn.ReLU(),
                                     torch.nn.Conv1d(in_channels=embed_dim, out_channels=256, kernel_size=1), torch.nn.ReLU(),
                                        torch.nn.Flatten(),  # RuntimeError: mat1 and mat2 shapes cannot be multiplied (16384x1 and 256x10)
                                                     torch.nn.Linear(256, args.num_classes))

    # classifier = torch.nn.Linear(embed_dim, args.num_classes)
    classifier.apply(segm_init_weights)

    proj_head = torch.nn.Sequential(torch.nn.Linear(embed_dim, 256),
                                    torch.nn.BatchNorm1d(256),
                                    torch.nn.ReLU(),
                                    torch.nn.Linear(256, 128))
    pred_head = torch.nn.Sequential(torch.nn.Linear(128, 256),
                                    torch.nn.BatchNorm1d(256),
                                    torch.nn.ReLU(),
                                    torch.nn.Linear(256, 128))
    # encoder.fc = torch.nn.Identity()

    encoder.cuda()
    classifier.cuda()
    proj_head.cuda()
    pred_head.cuda()


    criterion = torch.nn.CrossEntropyLoss()

    #################################### Training initialization #######################################
    optimizer = SGD([{'params': encoder.parameters()}, {'params': classifier.parameters()}, {'params': proj_head.parameters()}, {'params': pred_head.parameters()}],
                    lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr/50.0)

    acc_logs = open(f'{args.dataset}/{args.run_path}/acc.txt', 'w')
    stat_logs = open(f'{args.dataset}/{args.run_path}/stat.txt', 'w')
    save_config(args, f'{args.dataset}/{args.run_path}')
    print('Train args: \n', args)
    best_acc = 0

    with open("correct_before.txt", 'w') as correct_file:

        for idx in range(args.num_classes):
            # class_count = torch.sum(clean_label == idx).item()
            # clean_idx = torch.nonzero(clean_label == idx).squeeze().tolist()

            clean_idx = clean_label == idx
            class_count = torch.sum(clean_idx).item()
            class_right = torch.sum((clean_label == noisy_label) & clean_idx).item()

            correct_file.write( str(idx) + "类准确率为：" + str(float (class_right) / class_count) + "\n")
    
    ################################ Training loop ###########################################
    for i in range(args.epochs):

        clean_id, noisy_id, modified_label = evaluate(eval_loader, encoder, classifier, args, noisy_label, clean_label, i, stat_logs)
        # balanced_sampler
        clean_subset = Subset(train_data, clean_id.cpu())
        sampler = ClassBalancedSampler(labels=modified_label[clean_id], num_classes=args.num_classes)
        labeled_loader = torch.utils.data.DataLoader(clean_subset, batch_size=args.batch_size, sampler=sampler,
                                                     num_workers=4, drop_last=False)

        train(labeled_loader, modified_label, all_loader, encoder, classifier, proj_head, pred_head, optimizer, i, args)




        cur_acc = test(test_loader, encoder, classifier, i)
        scheduler.step()
        if cur_acc > best_acc:
            best_acc = cur_acc
            save_checkpoint({
                'cur_epoch': i,
                'classifier': classifier.state_dict(),
                'encoder': encoder.state_dict(),
                # 'proj_head': proj_head.state_dict(),
                # 'pred_head': pred_head.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, filename=f'{args.dataset}/{args.run_path}/best_acc.pth.tar')
        acc_logs.write(f'Epoch [{i}/{args.epochs}]: Best accuracy@{best_acc}! Current accuracy@{cur_acc} \n')
        acc_logs.flush()
        print(f'Epoch [{i}/{args.epochs}]: Best accuracy@{best_acc}! Current accuracy@{cur_acc} \n')


    save_checkpoint({
        'cur_epoch': args.epochs,
        'classifier': classifier.state_dict(),
        'encoder': encoder.state_dict(),
        'proj_head': proj_head.state_dict(),
        'pred_head': pred_head.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, filename=f'{args.dataset}/{args.run_path}/last.pth.tar')



if __name__ == '__main__':
    if os.path.exists("correct_before.txt"):
        os.remove("correct_before.txt")
    if os.path.exists("modified_calculate.txt"):
        os.remove("modified_calculate.txt")
    if os.path.exists("clean_select.txt"):
        os.remove("clean_select.txt")
    if os.path.exists("epoch_tp.txt"):
        os.remove("epoch_tp.txt")
    main()

