import os
import sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import torchvision
from torchvision import transforms
from torchsummary import summary

from models.resnet import ResNet18
from utils import *

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, '..'))


def run(args):
    device = torch.device('cuda')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)


    # ---- 화이트박스 워터마크 추출 설정 ------ #
    # binary prior info to be embedded, shape (T, 10) - 비트 b 로드
    b = np.load('logs/whitebox/marked/b.npy')
    # ---- 워터마크 추출 ------ #
    sd_path = 'logs/whitebox/ummarked/resnet18.pth'

    # ---- 화이트박스 워터마크 추출 (Alg. 3) ---- #
    marked_model = ResNet18().to(device)
    # summary(marked_model, input_size=(1, 28, 28))
    marked_model.load_state_dict(torch.load(sd_path))
    
    # subsample_training_data() - 키용 mini subset 생성 (Alg. 3 Step I)
    x_train_subset_loader = subsample_training_data(trainset, args.target_class)
    
    # get_activations() - 선택 층 출력 획득 (Alg. 3)
    marked_activations = get_activations(marked_model, x_train_subset_loader)
    print("Get activations of marked FC layer")
    # choose the activations from first wmarked dense layer
    marked_FC_activations = marked_activations
    
    # extract_WM_from_activations() - 워터마크 추출 (Alg. 3)
    A = np.load('logs/whitebox/marked/projection_matrix.npy')
    print('A = ', A)
    decoded_WM = extract_WM_from_activations(marked_FC_activations, A)
    
    # compute_BER() - 비트 오류율 계산 (Alg. 3 Step 5)
    BER = compute_BER(decoded_WM, b[:, args.target_class])
    print("BER in class {} is {}: ".format(args.target_class, BER))


def main():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--n_classes', type=int, default=10,
                        help='Number of classes in data')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--epochs', default=50, type=int, help='embed_epoch')
    parser.add_argument('--scale', default=0.01, type=float, help='for loss1')  # args.scale→λ₁
    parser.add_argument('--gamma', default=0.01, type=float, help='for loss2')  # args.gamma→λ₂
    parser.add_argument('--target_dense_idx', default=2, type=int, help='target layer to carry WM')
    parser.add_argument('--embed_bits', default=16, type=int)
    parser.add_argument('--target_class', default=0, type=int)
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
