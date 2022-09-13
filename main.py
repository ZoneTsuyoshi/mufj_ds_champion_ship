import os, shutil, argparse, subprocess, json


def main(config, debug=False, ensemble=False, mlm=False, pseudo_labeling=None, dirpath=None):
    if dirpath is None:
        dirpath = "exp/"
        if debug: dirpath += "d"
        if ensemble: dirpath += "e"
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
    with open(os.path.join(dirpath, "config.json"), "w") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)
    
    run_list = ["python"]
    if mlm:
        run_list.append(dirpath + "/run_mlm.py")
    elif ensemble:
        run_list.append(dirpath + "/run_ensemble.py")
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
    parser.add_argument("-e", action="store_true", help="ensemble")
    parser.add_argument("-m", action="store_true", help="mlm")
    parser.add_argument("-p", default=None, nargs="*", help="pseudo labeling, 1st: exp_id, 2nd: confidence")
    args = parser.parse_args()
    
    if args.p is None:
        f = open(args.config, "r")
    else:
        f = open(f"exp/{args.p[0]}/config.json", "r") 
    config = json.load(f)
    f.close()
    main(config, args.d, args.e, args.m, args.p)