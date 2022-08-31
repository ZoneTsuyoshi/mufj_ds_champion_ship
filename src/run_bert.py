import os, json, argparse
from train_bert import train
from test_bert import test

def main(debug=False):
    dirpath = os.path.dirname(__file__)
    train(dirpath, debug)
    test(dirpath, debug)
    
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='copy and run', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-d", action="store_true", help="debug")
    args = parser.parse_args()
    main(args.d)