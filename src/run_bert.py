import os, json, argparse, subprocess
# from train_bert import train
# from test_bert import test

def main(debug=False, pseudo_labeling=None):
    dirpath = os.path.dirname(__file__)
    train_list = ["python", dirpath + "/train_bert.py"]
    test_list = ["python", dirpath + "/test_bert.py"]
    if debug:
        train_list.append("-d")
        test_list.append("-d")
    if pseudo_labeling is not None:
        train_list += ["-p"] + pseudo_labeling
        
    subprocess.run(train_list)
    subprocess.run(test_list)
        
    # dirpath = os.path.dirname(__file__)
    # train(dirpath, debug, pseudo_labeling)
    # test(dirpath, debug)
    
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='copy and run', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-d", action="store_true", help="debug")
    parser.add_argument("-p", default=None, nargs="*", help="pseudo labeling, 1st: exp_id, 2nd: confidence")
    args = parser.parse_args()
    main(args.d, args.p)