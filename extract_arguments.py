"""run_infer.py

Usage:
  run_infer.py [options] [--help] <command> [<args>...]
  run_infer.py --version
  run_infer.py (-h | --help)

Options:
  -h --help                   Show this string.
  --version                   Show version.

  --gpu=<id>                  GPU list. [default: 0]
  --nr_types=<n>              Number of nuclei types to predict. [default: 0]
  --type_info_path=<path>     Path to a json define mapping between type id, type name, 
                              and expected overlaid color. [default: '']

  --model_path=<path>         Path to saved checkpoint.
  --model_mode=<mode>         Original HoVer-Net or the reduced version used PanNuke and MoNuSAC, 
                              'original' or 'fast'. [default: fast]
  --nr_inference_workers=<n>  Number of workers during inference. [default: 8]
  --nr_post_proc_workers=<n>  Number of workers during post-processing. [default: 16]
  --batch_size=<n>            Batch size per 1 GPU. [default: 32]

Two command mode are `tile` and `wsi` to enter corresponding inference mode
    tile  run the inference on tile
    wsi   run the inference on wsi

Use `run_infer.py <command> --help` to show their options and usage.
"""

tile_cli = """
Arguments for processing tiles.

usage:
    tile (--input_file=<path>) (--output_dir=<path>) \
         [--draw_dot] [--save_qupath] [--save_raw_map] [--mem_usage=<n>]
    
options:
   --input_file=<path>     Path to input data directory. Assumes the files are not nested within directory.
   --output_dir=<path>    Path to output directory..

   --mem_usage=<n>        Declare how much memory (physical + swap) should be used for caching. 
                          By default it will load as many tiles as possible till reaching the 
                          declared limit. [default: 0.2]
   --draw_dot             To draw nuclei centroid on overlay. [default: False]
   --save_qupath          To optionally output QuPath v0.2.3 compatible format. [default: False]
   --save_raw_map         To save raw prediction or not. [default: False]
"""

wsi_cli = """
Arguments for processing wsi

usage:
    wsi (--input_file=<path>) (--output_dir=<path>) [--proc_mag=<n>]\
        [--cache_path=<path>] [--input_mask_dir=<path>] \
        [--ambiguous_size=<n>] [--chunk_shape=<n>] [--tile_shape=<n>] \
        [--save_thumb] [--save_mask]
    
options:
    --input_file=<path>      Path to input data directory. Assumes the files are not nested within directory.
    --output_dir=<path>     Path to output directory.
    --cache_path=<path>     Path for cache. Should be placed on SSD with at least 100GB. [default: cache]
    --mask_dir=<path>       Path to directory containing tissue masks. 
                            Should have the same name as corresponding WSIs. [default: '']

    --proc_mag=<n>          Magnification level (objective power) used for WSI processing. [default: 40]
    --ambiguous_size=<int>  Define ambiguous region along tiling grid to perform re-post processing. [default: 128]
    --chunk_shape=<n>       Shape of chunk for processing. [default: 10000]
    --tile_shape=<n>        Shape of tiles for processing. [default: 2048]
    --save_thumb            To save thumb. [default: False]
    --save_mask             To save mask. [default: False]
"""


import logging
import os
import copy
from docopt import docopt
import ast

def get_arguments(args, sub_args, sub_cmd):

    # nr_gpus = torch.cuda.device_count()
    # log_info('Detect #GPUS: %d' % nr_gpus)
    nr_gpus = 1 # Change

    if args['model_path'] == None:
        raise Exception('A model path must be supplied as an argument with --model_path.')

    nr_types = int(args['nr_types']) if int(args['nr_types']) > 0 else None
    method_args = {
        'method' : {
            'model_args' : {
                'nr_types'   : nr_types,
                'mode'       : args['model_mode'],
            },
            'model_path' : args['model_path'],
        },
        'type_info_path'  : None if args['type_info_path'] == '' \
                            else args['type_info_path'],
    }

    # ***
    run_args = {
        'batch_size' : int(args['batch_size']) * nr_gpus,

        'nr_inference_workers' : int(args['nr_inference_workers']),
        'nr_post_proc_workers' : int(args['nr_post_proc_workers']),
    }

    if args['model_mode'] == 'fast':
        run_args['patch_input_shape'] = 256
        run_args['patch_output_shape'] = 164
    else:
        run_args['patch_input_shape'] = 270
        run_args['patch_output_shape'] = 80

    if sub_cmd == 'tile':
        run_args.update({
            'input_file'      : sub_args['input_file'],
            'output_dir'     : sub_args['output_dir'],

            'mem_usage'   : float(sub_args['mem_usage']),
            'draw_dot'    : sub_args['draw_dot'],
            'save_qupath' : sub_args['save_qupath'],
            'save_raw_map': sub_args['save_raw_map'],
        })

    if sub_cmd == 'wsi':
        run_args.update({
            'input_file'      : sub_args['input_file'],
            'output_dir'     : sub_args['output_dir'],
            'input_mask_dir' : sub_args['input_mask_dir'],
            'cache_path'     : sub_args['cache_path'],

            'proc_mag'       : int(sub_args['proc_mag']),
            'ambiguous_size' : int(sub_args['ambiguous_size']),
            'chunk_shape'    : int(sub_args['chunk_shape']),
            'tile_shape'     : int(sub_args['tile_shape']),
            'save_thumb'     : sub_args['save_thumb'],
            'save_mask'      : sub_args['save_mask'],
        })

    return method_args, run_args

def get_dict(hovernet_arguments_str, sub_cmd):

    args_str = "{" + hovernet_arguments_str.split("{")[1].split("}")[0] + "}"
    args_str = args_str.replace("'", '"')
    args = ast.literal_eval(args_str)

    sub_args_str = "{" + hovernet_arguments_str.split("{")[2].split("}")[0] + "}"
    sub_args_str = sub_args_str.replace("'", '"')
    sub_args = ast.literal_eval(sub_args_str)

    if(sub_cmd == "wsi"):

        if(args["nr_types"]):
            args["nr_types"] = int(args["nr_types"])
        if(args["nr_inference_workers"]):
            args["nr_inference_workers"] = int(args["nr_inference_workers"])
        if(args["nr_post_proc_workers"]):
            args["nr_post_proc_workers"] = int(args["nr_post_proc_workers"])
        if(args["batch_size"]):
            args["batch_size"] = int(args["batch_size"])

        if(sub_args["proc_mag"]):
            sub_args["proc_mag"] = int(sub_args["proc_mag"])
        if(sub_args["ambiguous_size"]):
            sub_args["ambiguous_size"] = int(sub_args["ambiguous_size"])
        if(sub_args["chunk_shape"]):
            sub_args["chunk_shape"] = int(sub_args["chunk_shape"])
        if(sub_args["tile_shape"]):
            sub_args["tile_shape"] = int(sub_args["tile_shape"])

    elif(sub_cmd == "tile"):
        
        if(args["nr_inference_workers"]):
            args["nr_inference_workers"] = int(args["nr_inference_workers"])
        if(args["nr_post_proc_workers"]):
            args["nr_post_proc_workers"] = int(args["nr_post_proc_workers"])
        if(args["batch_size"]):
            args["batch_size"] = int(args["batch_size"])
        if(args["nr_types"]):
            args["nr_types"] = int(args["nr_types"])

        if(sub_args["mem_usage"]):
            sub_args["mem_usage"] = float(sub_args["mem_usage"])

    return args, sub_args

def get_sub_cmd(hovernet_arguments_str):

    sub_cmd = hovernet_arguments_str.split("}")[-1].strip()
    return sub_cmd

if __name__ == '__main__':
    sub_cli_dict = {'tile' : tile_cli, 'wsi' : wsi_cli}
    arguments = docopt(__doc__, help=False, options_first=True, 
                    version='HoVer-Net Pytorch Inference v1.0')
    sub_cmd = arguments.pop('<command>')
    sub_cmd_arguments = arguments.pop('<args>')

    # ! TODO: where to save logging
    logging.basicConfig(
        level=logging.INFO,
        format='|%(asctime)s.%(msecs)03d| [%(levelname)s] %(message)s',datefmt='%Y-%m-%d|%H:%M:%S',
        handlers=[
            logging.FileHandler("debug.log"),
            logging.StreamHandler()
        ]
    )

    if arguments['--help'] and sub_cmd is not None:
        if sub_cmd in sub_cli_dict: 
            print(sub_cli_dict[sub_cmd])
        else:
            print(__doc__)
        exit()
    if arguments['--help'] or sub_cmd is None:
        print(__doc__)
        exit()

    sub_arguments = docopt(sub_cli_dict[sub_cmd], argv=sub_cmd_arguments, help=True)
    
    arguments.pop('--version')
    gpu_list = arguments.pop('--gpu')
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list

    # nr_gpus = torch.cuda.device_count()
    # log_info('Detect #GPUS: %d' % nr_gpus)
    nr_gpus = 1 # Change

    arguments = {k.replace('--', '') : v for k, v in arguments.items()}
    print(f"arguments :- {arguments}")
    sub_arguments = {k.replace('--', '') : v for k, v in sub_arguments.items()}
    print(f"sub_arguments :- {sub_arguments}")
    print(sub_cmd)

