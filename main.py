import argparse
from mnist import MNIST
from utils import parse_with_config
def main(args, BATCH_SIZE, EPOCH_NUM, SEED):
    mnist = MNIST(args, BATCH_SIZE, EPOCH_NUM, SEED)
    if args.mode == "train":
        mnist.train_model()
    elif args.mode == "test":
        mnist.test_model()
    elif args.mode == "predict":
        mnist.predict_model()
    else:
        raise OSError("Invalid mode. Please select from [train, test, predict]")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--batch_size", type=int, required=True, help="batch size")
    parser.add_argument("--epoch_num", type=int, required=True, help="number of epochs")
    parser.add_argument("--seed", type=int, default=123, help="random seed")
    parser.add_argument("--mode", type=str, required = True, help="train, test")
    parser.add_argument("--config", type=str, required = False, help="path to json file containing configuration")

    args = parse_with_config(parser)
    
    main(args, args.batch_size, args.epoch_num, args.seed)