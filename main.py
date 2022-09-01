import os, shutil, argparse, subprocess, json


def main(config, debug=False, ensemble=False, mlm=False, dirpath=None):
    if dirpath is None:
        dirpath = "exp/"
        if debug: dirpath += "d"
        if ensemble: dirpath += "e"
        if mlm: dirpath += "m"

        if debug:
            if os.path.exists(dirpath):
                shutil.rmtree(dirpath)
        else:
            i = 1
            while os.path.exists(dirpath + str(i)):
                i += 1
            dirpath += str(i)
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
    
    if debug:
        run_list.append("-d")
    subprocess.run(run_list)
    
    
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='copy and run', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-c", "--config", default="src/config.json", type=str, help="config")
    parser.add_argument("-d", action="store_true", help="debug")
    parser.add_argument("-e", action="store_true", help="ensemble")
    parser.add_argument("-m", action="store_true", help="mlm")
    args = parser.parse_args()
    
    f = open(args.config, "r")
    config = json.load(f)
    f.close()
    main(config, args.d, args.e, args.m)