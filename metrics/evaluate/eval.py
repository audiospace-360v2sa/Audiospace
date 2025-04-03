# navigate up one directory to get to stable-audio-metrics
import sys
import os

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# import packages
import pandas as pd
from src.passt_kld import passt_kld
from src.openl3_fd_stable import openl3_fd
from src.spatial import calculate_spatial_metrics
import argparse

parser = argparse.ArgumentParser(description='Compute metrics for the VGGSound dataset')
parser.add_argument('--generated-path', type=str, help='path with the audio to evaluate', required=True)
parser.add_argument('--csv-file-path', type=str,
                    help='file with ids and prompts correspondences', default=None)
parser.add_argument('--split-path', type=str, help='file with split', default=None)
parser.add_argument('--reference-path', type=str,
                    help='path with the recorded/reference/ground truth audio for FDopenl3 and KLpasst', required=True)
parser.add_argument('--kl-ref-prob', type=str, default=None, help='path to the reference probabilities for KLpasst')
parser.add_argument('--fd-ref-embeddings', type=str, default=None, help='path to the reference embeddings for FDopenl3')
parser.add_argument('--eval_files_extension', type=str, default=".flac", help='extension of the generated audio files')
parser.add_argument('--ref_files_extension', type=str, default=".flac", help='extension of the reference audio files')
args = parser.parse_args()

# the audio in generated_path should be named according to the 'ytid' csv_file_path below
generated_path = args.generated_path  # path with the audio to evaluate
csv_file_path = args.csv_file_path  # file with ids and prompts correspondences
split_path = args.split_path

# path with the recorded/reference/ground truth audio for FDopenl3 and KLpasst
reference_path = args.reference_path

#  these are the musiccaps ids that we could not download from Youtube – we ignore them for KLpasst computation
# at the time of downloading musiccaps, 5434 out of 5521 audios were available, this is the list of audios that were not available:
# NOT_IN_MUSICCAPS = ['NXuB3ZEpM5U', 'C7OIuhWSbjU', 'Rqv5fu1PCXA', 'WvEtOYCShfM', '25Ccp8usBtE', 'idVCpQaByc4', 'tpamd6BKYU4', 'bpwO0lbJPF4', 'We0WIPYrtRE', 'kiu-40_T5nY', '5Y_mT93tkvQ', 'zCrpaLEq1VQ', '8olHAhUKkuk', '6xxu6f0f0e4', 'B7iRvj8y9aU', 'rrAIuGMTqtA', 'UdA6I_tXVHE', 'm-e3w2yZ6sM', 'Xy7KtmHMQzU', 'd6-bQMCz7j0', 'BeFzozm_H5M', 't5fW1-6iXZY', 'jd1IS7N3u0I', '_hYBs0xee9Y', 'EhFWLbNBOxc', '63rqIYPHvlc', 'Jk2mvFrdZTU', 'IbJh1xeBFcI', 'HAHn_zB47ig', 'j9hAUlz5kQs', 'Vu7ZUUl4VPc', 'asYb6iDz_kM', 'fZyq2pM2-dI', 'vOAXAoHtl7o', 'go_7i6WvfeE', 'iXgEQj1Fs7g', 'dcY062mkf9g', '_ACyiKGpD8Y', '_DHMdtRRJzE', 'zSSIGv82318', '2dyxjGTXSpA', '7WZwlOrRELI', 'g8USMvt9np0', '374R7te0ra0', 'CCFYOw8keiI', 'eHeUipPZHIc', '0J_2K1Gvruk', 'MYtq46rNsCA', 'NIcsJ8sEd0M', '8vFJX7NcSbI', 'TkclVqlyKx4', 'T6iv9GFIVyU', 'ChqJYrmQIN4', 'ZZrvO__SNtA', 'fwXh_lMOqu0', '0khKvVDyYV4', '-sevczF5etI', 'qc1DaM4kdO0', 'wBe5tW8iJew', 'vQHKa69Mkzo', 'Fv9swdLA-lo', 'Ah_aYOGnQ_I', 'nTtxF9Wyw6o', '7B1OAtD_VIA', 'OS4YFp3DiEE', 'lTLsL94ABRs', 'jmPmqzxlOTY', 'k-LkhT4HAiE', 'Hvs6Xwc6-gc', 'xxCnmao8FAs', 'BiQik0xsWxk', 'L5Uu_0xEZg4', 'cADT8fUucLQ', 'ed-1zAOr9PQ', 'zSq2D_GF00o', 'gdtw54I8soM', 'lrk00BNiuD4', 'RQ0-sjpAPKU', 'SLq-Co_szYo', '0fqtA_ZBn_8', 'Xoke1wUwEXY', 'LRfVQsnaVQE', 'p_-lKpxLK3g', 'AaUZb-iRStE', '0pewITE1550', 'JNw0A8pRnsQ', 'vVNWjq9byoQ']
NOT_IN_MUSICCAPS = []

musiccaps_ids = []
if csv_file_path is not None:
    df = pd.read_csv(csv_file_path)
    musiccaps_ids = df['ytid'].tolist()
    print('csv_file_path:', csv_file_path)
elif split_path is not None:
    with open(split_path, 'r') as f:
        for line in f:
            musiccaps_ids.append(os.path.splitext(line.strip())[0])
    print('split_path:', split_path)
else:
    raise ValueError('Either csv_file_path or split_path must be provided')

# print('Computing CLAP score..')
# # in this musiccaps case here, our audios are named with the ytid in csv_file_path
# # create a dictionary to get the text prompt (used to generate audio) given the ytid (audio file name)
# id2text = df.set_index('ytid')['caption'].to_dict()
# # compute clap score from the id2text (prompts) and generated_path (audio)
# clp = clap_score(id2text, generated_path, audio_files_extension='.wav')
# print('[musiccaps] CLAP score (630k-audioset-fusion-best.pt): ', clp, generated_path)


print('Computing KLpasst..')
# list all ids that are in both ref_path (reference audio) and eval_path (generated audio)
# in this musiccaps case here, our audios are named with the ytid in csv_file_path
# musiccaps_ids = df['ytid'].tolist()

# compute KLpasst between ref_path (reference audio) and eval_path (generated audio)
kl = passt_kld(ids=musiccaps_ids,
               eval_path=generated_path,
               ref_path=reference_path,
               eval_files_extension=args.eval_files_extension,
               ref_files_extension=args.ref_files_extension,
               no_ids=NOT_IN_MUSICCAPS,
               collect='mean',
               load_ref_probabilities=args.kl_ref_prob)
print('KLpasst: ', kl, generated_path)

print('Computing FDopenl3..')
model_channels = 2  #  1 or 2 channels
model_sr = 44100  #  maximum bandwidth at which we evaluate, up to 48kHz
type = 'env'  #  openl3 model trained on 'music' or 'env'
hop = 0.5  # openl3 hop_size in seconds (openl3 window is 1 sec)
batch = False
# compute the FDopenl3 given the parameters above
fd = openl3_fd(
    ids=musiccaps_ids,
    channels=model_channels,
    samplingrate=model_sr,
    content_type=type,
    openl3_hop_size=hop,
    eval_path=generated_path,
    eval_files_extension=args.eval_files_extension,
    ref_path=reference_path,
    ref_files_extension=args.ref_files_extension,
    batching=batch,
    load_ref_embeddings=args.fd_ref_embeddings
)

print('FDopenl3: ', fd, generated_path)

print('Computing spatial metrics..')
# compute the spatial metrics
avg_theta, avg_phi, avg_spatial = calculate_spatial_metrics(
    reference_path,
    generated_path,
    split_path,
    error_type="MAE"
)

print('KLpasst: ', kl)
print('FDopenl3: ', fd)
print('Spatial metrics: ')
print('Theta: ', avg_theta)
print('Phi: ', avg_phi)
print('Spatial angle: ', avg_spatial)
print('All metrics computed!')
