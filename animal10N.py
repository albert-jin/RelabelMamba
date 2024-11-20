import argparse

import torch.optim.lr_scheduler
import torchvision.transforms as transforms
import wandb

from torch.optim import SGD
from torch.utils.data import Subset
from tqdm import tqdm
import torch.nn as nn

from datasets.dataloader_animal10n import animal_dataset
from torchvision.models import vgg19_bn
from utils import *
from vim.models_mamba import segm_init_weights, \
    vim_small_patch16_stride8_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2

parser = argparse.ArgumentParser('Train with ANIMAL-10N dataset')
parser.add_argument('--dataset_path', default='/data/data/academic/ANIMAL-10N', help='dataset path')

# model settings
parser.add_argument('--theta_s', default=1.0, type=float, help='threshold for voted correct samples (default: 1.0)')
parser.add_argument('--theta_r', default=0.95, type=float, help='threshold for relabel samples (default: 0.95)')
parser.add_argument('--k', default=200, type=int, help='neighbors for soft-voting (default: 200)')

# train settings
parser.add_argument('--epochs', default=30, type=int, metavar='N', help='number of total epochs to run (default: 200)')
parser.add_argument('--batch_size', default=25, type=int, help='mini-batch size (default: 128)')
parser.add_argument('--lr', default=0.0002, type=float, help='initial learning rate (default: 0.02)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum of SGD solver (default: 0.9)')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay (default: 5e-4)')
parser.add_argument('--seed', default=3047, type=int, help='seed for initializing training. (default: 3047)')
parser.add_argument('--gpuid', default='0', type=str, help='Selected GPU (default: "0")')
parser.add_argument('--entity', type=str, help='Wandb user entity')
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

def train(labeled_trainloader, modified_label, all_trainloader, encoder, classifier, optimizer, epoch, args):
    encoder.eval()
    classifier.train()
    xlosses = AverageMeter('xloss')
    labeled_train_iter = iter(labeled_trainloader)
    all_bar = tqdm(all_trainloader)
    for [inputs_x1, inputs_x2], labels_x, index in labeled_train_iter:
        # cross-entropy training with mixup
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

        # mixed_input = l * input_a + (1 - l) * input_b
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
    # logger.log({'ce loss': xlosses.avg, 'fc loss': ulosses.avg})


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
            pred = torch.argmax(res, dim=1)
            acc = torch.sum(pred == label) / float(data.size(0))
            accuracy.update(acc.item(), data.size(0))
            data_bar.set_description(f'Test epoch {epoch}: Accuracy#{accuracy.avg:.4f}')
    # logger.log({'acc': accuracy.avg})
    return accuracy.avg


def evaluate(dataloader, encoder, classifier, args, noisy_label, i):
    encoder.eval()
    classifier.eval()
    prediction = []
    feature_arrbank = []

    ################################### feature extraction ###################################
    with torch.no_grad():
        # generate feature bank
        for (data, target, index) in tqdm(dataloader, desc='Feature extracting'):
            data = data.cuda()
            # feature = encoder(data)
            feature = encoder(data, is_evaluate = True)
            feature_arr = feature.cpu().numpy()
            feature_arrbank.append(feature_arr)
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
        conf_id = torch.where(his_score > args.theta_r)[0]
        modified_label = torch.clone(noisy_label).detach()
        modified_label[conf_id] = his_label[conf_id]

        ################################### sample selection ###################################
        prediction_knn = weighted_knn(feature_bank, feature_bank, modified_label, args.num_classes, args.k, 10)  # temperature in weighted KNN
        vote_y = torch.gather(prediction_knn, 1, modified_label.view(-1, 1)).squeeze()
        vote_max = prediction_knn.max(dim=1)[0]
        right_score = vote_y / vote_max
        clean_id = torch.where(right_score >= args.theta_s)[0]
        noisy_id = torch.where(right_score < args.theta_s)[0]

    return clean_id, noisy_id, modified_label


def main():
    args = parser.parse_args()
    seed_everything(args.seed)
    if args.run_path is None:
        args.run_path = f'Dataset(animal10n_Model({args.theta_s})'

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid
    # os.environ['WANDB_API_KEY'] = "21e471e150b8b942b9032ee23b4ebe31c2eb85ee"
    # os.environ['WANDB_MODE'] = "offline"
    global logger
    logger = wandb.init(project='animal10n', entity=args.entity, name=args.run_path)
    logger.config.update(args)

    if not os.path.isdir(f'animal10n'):
        os.mkdir(f'animal10n')
    if not os.path.isdir(f'animal10n/{args.run_path}'):
        os.mkdir(f'animal10n/{args.run_path}')

    ############################# Dataset initialization ##############################################
    args.num_classes = 10
    args.image_size = 224  # 64
    image_dimension = 224
    target_size = (image_dimension, image_dimension)

    # data loading
    weak_transform = transforms.Compose([
        ResizeAndPad(target_size, 14),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()])
    none_transform = transforms.Compose([
        ResizeAndPad(target_size, 14),
        transforms.ToTensor()])  # no augmentation
    strong_transform = transforms.Compose([
        # transforms.RandomCrop(args.image_size, padding=8),
        ResizeAndPad(target_size, 14),
        transforms.RandomHorizontalFlip(),
        RandAugment(),
        transforms.ToTensor()])

    # eval data served as soft-voting pool
    train_data = animal_dataset(root=args.dataset_path, transform=KCropsTransform(strong_transform, 2), mode='train')
    eval_data = animal_dataset(root=args.dataset_path, transform=weak_transform, mode='train')
    test_data = animal_dataset(root=args.dataset_path, transform=none_transform, mode='test')
    all_data = animal_dataset(root=args.dataset_path, transform=MixTransform(strong_transform, weak_transform, 1), mode='train')

    # noisy labels
    noisy_label = torch.tensor(eval_data.targets).cuda()

    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)  # num_workers=4
    eval_loader = torch.utils.data.DataLoader(eval_data, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    all_loader = torch.utils.data.DataLoader(all_data, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    ################################ Model initialization ###########################################
    encoder = vim_small_patch16_stride8_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2(
        pretrained=False, num_classes=args.num_classes, drop_rate=0.0, drop_path_rate=0.1,
        drop_block_rate=None, img_size=args.image_size)
    checkpoint = torch.load("vim/pretrained/vim_s_midclstok_ft_81p6acc.pth",
                            map_location='cpu')  # vim_s_midclstok_80p5acc
    checkpoint_model = checkpoint['model']

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
    embed_dim = 384
    classifier = torch.nn.Sequential(
        torch.nn.Conv1d(in_channels=embed_dim, out_channels=256, kernel_size=1), torch.nn.ReLU(),
        torch.nn.Flatten(),
        torch.nn.Linear(256, args.num_classes))
    classifier.apply(segm_init_weights)



    encoder.cuda()
    classifier.cuda()

    #################################### Training initialization #######################################
    optimizer = SGD([{'params': encoder.parameters()}, {'params': classifier.parameters()}],
                    lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100], gamma=0.1)

    acc_logs = open(f'animal10n/{args.run_path}/acc.txt', 'w')
    save_config(args, f'animal10n/{args.run_path}')
    print('Train args: \n', args)
    best_acc = 0

    ################################ Training loop ###########################################
    for i in range(args.epochs):  # args.epochs
        clean_id, noisy_id, modified_label = evaluate(eval_loader, encoder, classifier, args, noisy_label, i)

        clean_subset = Subset(train_data, clean_id.cpu())
        sampler = ClassBalancedSampler(labels=modified_label[clean_id], num_classes=args.num_classes)
        labeled_loader = torch.utils.data.DataLoader(clean_subset, batch_size=args.batch_size, sampler=sampler, num_workers=4, drop_last=True)  # num_workers=4

        train(labeled_loader, modified_label, all_loader, encoder, classifier, optimizer, i, args)

        cur_acc = test(test_loader, encoder, classifier, i)
        scheduler.step()
        if cur_acc > best_acc:
            best_acc = cur_acc
            save_checkpoint({
                'cur_epoch': i,
                'classifier': classifier.state_dict(),
                'encoder': encoder.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, filename=f'animal10n/{args.run_path}/best_acc.pth.tar')
        acc_logs.write(f'Epoch [{i}/{args.epochs}]: Best accuracy@{best_acc}! Current accuracy@{cur_acc} \n')
        acc_logs.flush()
        print(f'Epoch [{i}/{args.epochs}]: Best accuracy@{best_acc}! Current accuracy@{cur_acc} \n')
    save_checkpoint({
        'cur_epoch': i,
        'classifier': classifier.state_dict(),
        'encoder': encoder.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, filename=f'animal10n/{args.run_path}/last.pth.tar')


if __name__ == '__main__':
    main()
