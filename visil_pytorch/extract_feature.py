import json
import torch
import argparse
import pickle
from tqdm import tqdm
import sys 
import gc
sys.path.append("/mnt/big-data/intern/son/Master/visil_pytorch")
from model.visil import ViSiL
from torch.utils.data import DataLoader
from datasets.generators import VideoGenerator
from evaluation import extract_features


def process_batch(frames):
    with torch.no_grad():
        # Process your batch here
        features = extract_features(model, frames, args)
        return features

if __name__ == '__main__':
    formatter = lambda prog: argparse.ArgumentDefaultsHelpFormatter(prog, max_help_position=80)
    parser = argparse.ArgumentParser(description='This is the code for video similarity calculation based on ViSiL network.', formatter_class=formatter)
    parser.add_argument('--database_file', type=str, required=True,
                        help='Path to file that contains the database videos')
    parser.add_argument('--output_file', type=str, default='features_store.pkl',
                        help='Name of the output file.')
    parser.add_argument('--batch_sz', type=int, default=128,
                        help='Number of frames contained in each batch during feature extraction. Default: 128')
    parser.add_argument('--batch_sz_sim', type=int, default=2048,
                        help='Number of feature tensors in each batch during similarity calculation.')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='Id of the GPU used.')
    parser.add_argument('--load_queries', action='store_true',
                        help='Flag that indicates that the queries will be loaded to the GPU memory.')
    parser.add_argument('--similarity_function', type=str, default='chamfer', choices=["chamfer", "symmetric_chamfer"],
                        help='Function that will be used to calculate similarity '
                             'between query-target frames and videos.')
    parser.add_argument('--workers', type=int, default=16,
                        help='Number of workers used for video loading.')
    args = parser.parse_args()

    # Initialize ViSiL model
    model = ViSiL(pretrained=True, symmetric='symmetric' in args.similarity_function).to(args.gpu_id)
    model.eval()
    # Create a video generator for the database video
    generator = VideoGenerator(args.database_file)
    loader = DataLoader(generator, num_workers=args.workers)

    pbar = tqdm(loader)
    features_store = {}
    print('\n> Calculate database features')

    for video in pbar:
        frames = video[0][0]
        video_id = video[1][0]
        if frames.shape[0] > 1:
            features = process_batch(frames).detach().cpu()
            features_store[video_id] = features
        torch.cuda.empty_cache()
        gc.collect()
    with open(args.output_file, 'wb') as file:
        pickle.dump(features_store, file)

 