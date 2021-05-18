import argparse

def parse_opt():
    parser = argparse.ArgumentParser()
    # Data input settings
    parser.add_argument('--rnn_size', type=int, default=1280,
                        help='size of the rnn in number of hidden nodes in question gru')
    parser.add_argument('--num_hid', type=int, default=1280,
                        help='size of the rnn in number of hidden nodes in question gru')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of GCN layers')
    parser.add_argument('--rnn_type', type=str, default='gru',
                        help='rnn, gru, or lstm')
    parser.add_argument('--v_dim', type=int, default=2048,
                        help='2048 for resnet, 4096 for vgg')
    parser.add_argument('--ans_dim', type=int, default=2274,
                        help='3219 for VQA-CP v2, 2185 for VQA-CP v1')
    parser.add_argument('--logit_layers', type=int, default=1,
                        help='number of layers in the RNN')
    parser.add_argument('--activation', type=str, default='ReLU',
                        help='number of layers in the RNN')
    parser.add_argument('--norm', type=str, default='weight',
                        help='number of layers in the RNN')
    parser.add_argument('--initializer', type=str, default='kaiming_normal',
                        help='number of layers in the RNN')

    # Optimization: General
    parser.add_argument('--num_epochs', type=int, default=20,
                    help='number of epochs')
    parser.add_argument('--train_candi_ans_num', type=int, default=20,
                    help='number of candidate answers')

    parser.add_argument('--s_epoch', type=int, default=0,
                    help='training from s epochs')
    parser.add_argument('--ratio', type=float, default=1,
                    help='ratio of training set used')
    parser.add_argument('--batch_size', type=int, default=32,
                    help='minibatch size')
    parser.add_argument('--grad_clip', type=float, default=0.25,
                    help='clip gradients at this value')
    parser.add_argument('--dropC', type=float, default=0.5,
                    help='strength of dropout in the Language Model RNN')
    parser.add_argument('--dropG', type=float, default=0.2,
                    help='strength of dropout in the Language Model RNN')
    parser.add_argument('--dropL', type=float, default=0.1,
                    help='strength of dropout in the Language Model RNN')
    parser.add_argument('--dropW', type=float, default=0.4,
                    help='strength of dropout in the Language Model RNN')
    parser.add_argument('--dropout', type=float, default=0.2,
                    help='strength of dropout in the Language Model RNN')

    #Optimization: for the Language Model
    parser.add_argument('--optimizer', type=str, default='adam',
                    help='what update to use? rmsprop|sgd|sgdmom|adagrad|adam')
    parser.add_argument('--learning_rate', type=float, default=1e-5,
                    help='learning rate')
    parser.add_argument('--self_loss_weight', type=float, default=3,
                    help='self-supervised loss weight')
    parser.add_argument('--self_sup', type=int, default=1,
                    help='whether using self-sup processing')
    parser.add_argument('--optim_alpha', type=float, default=0.9,
                    help='alpha for adam')
    parser.add_argument('--optim_beta', type=float, default=0.999,
                    help='beta used for adam')
    parser.add_argument('--optim_epsilon', type=float, default=1e-8,
                    help='epsilon that goes into denominator for smoothing')
    parser.add_argument('--weight_decay', type=float, default=0,
                    help='weight_decay')
    parser.add_argument('--seed', type=int, default=1024,
                    help='seed')
    parser.add_argument('--ntokens', type=int, default=777,
                    help='ntokens')
    
    parser.add_argument('--dataroot', type=str, default='../../SSL-VQA/data/vqacp2/',help='dataroot')
    parser.add_argument('--img_root', type=str, default='../../SSL-VQA/data/coco/',help='image_root')
                    
    parser.add_argument('--checkpoint_path4test', type=str, default='saved_models_cp2/base/SAR_LMH_top20_best_model.pth',
                    help='directory to store checkpointed models4test, used for testing')
    parser.add_argument('--checkpoint_path4test_QTDmodel', type=str, default='data4VE/offline-QTD_model.pth',
                    help='directory to store the QTDmodel, used for testing')
    parser.add_argument('--test_type', type=str, default='SAR_Top20',
                    help='name of saved model')
    parser.add_argument('--lp', type=int, default=0, #[0, 1, 2]
                    help='the combination with Language-Priors method: 0-Non_LP; 1-SSL; 2-LMH')
    parser.add_argument('--test_candi_ans_num', type=int, default=12,
                    help='number of candidate answers in test')
    parser.add_argument('--QTD_N4yesno', type=int, default=1,
                    help='number for the candidate answers of yes/no question in test')
    parser.add_argument('--QTD_N4non_yesno', type=int, default=12,
                    help='number for the candidate answers of non-yes/no question in test')



    parser.add_argument('--input', type=str, default=None)
    parser.add_argument('--output', type=str, default='saved_models_cp2/base/')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--logits', action='store_true')
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--label', type=str, default='best')

    args = parser.parse_args()

    return args
