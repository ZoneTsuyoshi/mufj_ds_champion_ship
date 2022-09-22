import os, shutil, argparse, subprocess, json


def main(config, debug=False, ensemble=None, mlm=False, pseudo_labeling=None, dirpath=None, gpu=None, epoch=None, ensemble_method="Nelder-Mead", ensemble_loss="f1", ensemble_threshold_search=False):
    if dirpath is None:
        dirpath = "exp/"
        if debug: dirpath += "d"
        if ensemble is not None: dirpath += "e"
        if mlm: dirpath += "m"
        if pseudo_labeling is not None:
            dirpath += pseudo_labeling[0] + "p"
            
        if not debug:
            i = 1
            while os.path.exists(dirpath + str(i)):
                i += 1
            dirpath += str(i)
    if debug and os.path.exists(dirpath):
        shutil.rmtree(dirpath)
    print("set directory {}".format(dirpath))
    
    shutil.copytree("src", dirpath)
    config["exp"]["dirpath"] = dirpath
    if gpu is not None:
        config["train"]["gpu"] = int(gpu)
    if epoch is not None:
        config["train"]["epoch"] = int(gpu)
    with open(os.path.join(dirpath, "config.json"), "w") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)
    
    run_list = ["python"]
    if mlm:
        run_list.append(dirpath + "/run_mlm.py")
    elif ensemble is not None:
        run_list += [dirpath + "/run_ensemble.py", "-e"] + ensemble + ["-m", ensemble_method, "-l", ensemble_loss, "-t", ensemble_threshold_search]
    else:
        run_list.append(dirpath + "/run_bert.py")
        if pseudo_labeling is not None:
            run_list += ["-p"] + pseudo_labeling
    
    if debug:
        run_list.append("-d")
    subprocess.run(run_list)
    
    
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='copy and run', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", default="src/config.json", type=str, help="config")
    parser.add_argument("-d", action="store_true", help="debug")
    parser.add_argument("-e", nargs="*", default=None, help="ensemble")
    parser.add_argument("-m", action="store_true", help="mlm")
    parser.add_argument("-p", default=None, nargs="*", help="pseudo labeling, 1st: exp_id, 2nd: confidence")
    parser.add_argument("-g", type=int, default=None, help="gpu id")
    parser.add_argument("-ep", type=int, default=None, help="epoch")
    
    parser.add_argument("-em", type=str, default="Nelder-Mead", help="ensemble method")
    parser.add_argument("-el", type=str, default="f1", help="ensemble loss")
    parser.add_argument("-et", action="store_true", help="threshold search for ensemble")
    args = parser.parse_args()
    
    if args.p is None:
        f = open(args.config, "r")
    else:
        f = open(f"exp/{args.p[0]}/config.json", "r") 
    config = json.load(f)
    f.close()
    main(config, args.d, args.e, args.m, args.p, None, args.g, args.ep, args.em, args.el, args.et)