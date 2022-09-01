import os, json, time, shutil, copy, subprocess, argparse
import multiprocessing as mp
import numpy as np

    
def gs_main(debug=False, mlm=False, parallel_strategy_on=False, max_parallel_queues=3, minimum_memory=1500):
    f = open("src/config.json", "r")
    config = json.load(f)
    f.close()
    
    flatten_config = {}
    for key in config.keys():
        flatten_config.update(config[key])

    gpu_id = config["train"]["gpu"]
    model_dict = {"xlm-roberta-base":{"bs":16},
                 "xlm-roberta-large":{"bs":8},
                 "bert-base-multilingual-uncased":{"bs":16}}
    model_list = ["xlm-roberta-base", "xlm-roberta-large", "bert-base-multilingual-uncased"]
    bs_list = [model_dict[m]["bs"] for m in model_list]
    gs_dict = {"mix":{"model":model_list, "batch_size":bs_list, "gpu":[0,1,2]}}


    gs_key = list(gs_dict.keys()) # list of keys for grid search
    gs_length = len(gs_dict)
    gs_key2 = []
    for key in gs_key:
        # if dictionary has hierarchical structure, add hierarchical keys to gs_key2 list.
        if type(gs_dict[key])==list:
            gs_key2.append(key)
        elif type(gs_dict[key])==dict:
            gs_key2 += list(gs_dict[key].keys())
    
    dirpath = "exp/"
    if debug: dirpath += "d"
    if mlm: dirpath += "m"

    start_id = 1
    if not debug:
        while os.path.exists(dirpath + str(start_id)):
            start_id += 1

    config_list = []
        
        
    def generate_queue_flatten_config(old_config, depth):
        key = gs_key[depth]
        
        if type(gs_dict[key])==list:
            for i, value in enumerate(gs_dict[key]):
                new_config = copy.deepcopy(old_config)
                new_config[key] = value
                if depth+1 < gs_length:
                    generate_queue_flatten_config(new_config, depth+1)
                else:
                    config_list.append(new_config)
        elif type(gs_dict[key])==dict:
            interlocking_key = list(gs_dict[key].keys())
            min_length = 100
            for ikey in interlocking_key:
                min_length = len(gs_dict[key][ikey]) if len(gs_dict[key][ikey]) < min_length else min_length
            for i in range(min_length):
                new_config = copy.deepcopy(old_config)
                for ikey in interlocking_key:
                    new_config[ikey] = gs_dict[key][ikey][i]
                if depth+1 < gs_length:
                    generate_queue_flatten_config(new_config, depth+1)
                else:
                    config_list.append(new_config)
        else:
            raise ValueError("elements must be a list type object or a dict type object")
            
    
    def flatten_config_to_parse(config):
        parse_list = []
        for key in config.keys():
            parse_list.append("--{}".format(key))
            if type(config[key])==list:
                parse_list += [str(value) for value in config[key]]
            else:
                parse_list.append(str(config[key]))
        return parse_list
    
            
    
    generate_queue_flatten_config(flatten_config, 0)
    total_parse_list = []
    for i, config_element in enumerate(config_list):
        config_element["dirpath"] = dirpath + str(start_id + i)
        parse_element = ["python", "parse.py"] + flatten_config_to_parse(config_element)
        if debug: parse_element.append("-d")
        if mlm: parse_element.append("-m")
        total_parse_list.append(parse_element)
    
    
    if parallel_strategy_on:
        for i in range((len(config_list)-1)//max_parallel_queues+1):
            if "gpu" in gs_key:
                gpu_ids = gs_dict["gpu"]
                memory_used = [int(get_gpu_info()[gpu_id]["memory.used"]) for gpu_id in gpu_ids]
                while max(memory_used) > minimum_memory:
                    print("waiting in {}-th parallel computation".format(i+1))
                    time.sleep(10)
                    memory_used = [int(get_gpu_info()[gpu_id]["memory.used"]) for gpu_id in gpu_ids]
            elif type(gpu_id)==int:
                memory_used = int(get_gpu_info()[gpu_id]["memory.used"])
                while memory_used > minimum_memory:
                    print("waiting in {}-th parallel computation".format(i+1))
                    time.sleep(10)
                    memory_used = int(get_gpu_info()[gpu_id]["memory.used"])
            p = mp.Pool(min(mp.cpu_count(), max_parallel_queues))
            p.map(subprocess.run, total_parse_list[max_parallel_queues*i:max_parallel_queues*(i+1)])
            p.close()
    else:
        for parse_element in total_parse_list:
            subprocess.run(parse_element)
    
    
            
def get_gpu_info(nvidia_smi_path='nvidia-smi', keys=("index", "uuid", "name", "timestamp", "memory.total", "memory.free", "memory.used", "utilization.gpu", "utilization.memory"), no_units=True):
    nu_opt = '' if not no_units else ',nounits'
    cmd = '%s --query-gpu=%s --format=csv,noheader%s' % (nvidia_smi_path, ','.join(keys), nu_opt)
    output = subprocess.check_output(cmd, shell=True)
    lines = output.decode().split('\n')
    lines = [ line.strip() for line in lines if line.strip() != '' ]

    return [ { k: v for k, v in zip(keys, line.split(', ')) } for line in lines ]
        
        
        
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Main Routine for Swithing Trajectory Network', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-d", action="store_true", help="debug")
    parser.add_argument("-m", action="store_true", help="mlm")
    parser.add_argument("-s", action="store_true", help="not parallel")
    parser.add_argument("-q", "--queue", type=int, default=3, help="max queue")
    parser.add_argument("-M", "--memory", type=int, default=2000, help="minimum memory (store)")
    args = parser.parse_args()
    
    gs_main(args.d, args.m, not args.s, args.queue, args.memory)