import json
import torch
import argparse
import pickle
from tqdm import tqdm
from model.visil import ViSiL
from torch.utils.data import DataLoader
from datasets.generators import VideoGenerator, VideoGeneratorForInference
from evaluation import extract_features, calculate_similarities_to_queries
import yt_dlp

import mmap
def download_video(args, video_id):
    """
      Download the provided video using the yt-dlp library

      Args:
        video_id: Youtube ID of the video
        args: arguments provided by the user
      Returns:
        a flag that indicates whether there was an error during the downloading
    """

    ydl_opts = {
            'format': 'best[height<={}][ext=mp4]/best[ext=mp4]/best[height<={}]/best'
                .format("480", "480"),
            'outtmpl': '{}/{}.%(ext)s'.format(args.video_dir, video_id),
            'quiet': True,
            'no_warnings': True
        }
    print(ydl_opts)
    ydl = yt_dlp.YoutubeDL(ydl_opts)
    video_url = 'https://www.youtube.com/watch?v={}'.format(video_id)
    ydl.download([video_url])

    with open(args.query_file, 'w') as f:
        line = video_id + "\t" + args.video_dir + "/" + video_id + ".mp4"
        f.write(line+"\n")


if __name__ == '__main__':
    formatter = lambda prog: argparse.ArgumentDefaultsHelpFormatter(prog, max_help_position=80)
    parser = argparse.ArgumentParser(description='This is the code for video similarity calculation based on ViSiL network.', formatter_class=formatter)
    parser.add_argument('--query_file', type=str, required=True,
                        help='Path to file that contains the query videos')
    parser.add_argument('--feature_file', type=str, required=True,
                        help='Path to file that contains the query videos')
    parser.add_argument('--video_id', type=str, default='-08oAPmbXx8',
                        help='video on youtube, ex: https://www.youtube.com/watch?v=swPJI3tAU20 .')
    parser.add_argument('--output_file', type=str, default='results.json',
                        help='Name of the output file.')
    parser.add_argument('--video_dir', type=str, default='results.json',
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

    # Create a video generator for the queries
    download_video(args,video_id=args.video_id)
    generator = VideoGenerator(args.query_file)
    loader = DataLoader(generator, num_workers=args.workers)
    generator_feature = VideoGeneratorForInference(args.feature_file)
    feature = DataLoader(generator_feature, num_workers=args.workers)

    # Initialize ViSiL model
    model = ViSiL(pretrained=True, symmetric='symmetric' in args.similarity_function).to(args.gpu_id)
    model.eval()

    # # Extract features of the queries
    queries, queries_ids = [], []
    pbar = tqdm(loader)

    print('> Extract features of the query videos')
    for video in pbar:
        frames = video[0][0]
        video_id = video[1][0]
        features = extract_features(model, frames, args)
        if not args.load_queries: features = features.cpu()
        queries.append(features)
        queries_ids.append(video_id)
        pbar.set_postfix(query_id=video_id)

    # feature = pickle.load(open(args.feature_file,'rb'))

    similarities = dict({query: dict() for query in queries_ids})
    
    print('\n> Calculate query-target similarities')
    for video,id in tqdm(feature):
        sims = calculate_similarities_to_queries(model, queries, video.squeeze(0).to(args.gpu_id), args)
        for i, s in enumerate(sims):
            similarities[queries_ids[i]]['https://www.youtube.com/watch?v={}'.format(id[0])] = float(s)

    sorted_item = sorted(similarities[queries_ids[0]].items(), key=lambda k: k[1], reverse=True)
    print(dict(sorted_item[:5]))

