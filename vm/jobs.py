"""Writes the pfetch job list of one download (backbone shards, Engram or DSpark draft) at pinned HF revisions. usage: python3 jobs.py backbone|engram|draft DEST_DIR OUT.json"""
import json, os, sys

HF = 'https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF/resolve/a419d74cd28d5b9f8c5e2b9077d11a609c17bc32/'
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models.json')) as f:
    models = json.load(f)
model = models[os.environ.get('MODEL_VARIANT', 'original')]
NAME = 'DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-%05d-of-00006.gguf'
SIZES = model['sizes']
ENGRAM = 'DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf'
DRAFT = 'https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF/resolve/85f81fb970add013033f25e94006713ca2c0aaf2/dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf'

what, dest, out = sys.argv[1:4]
files = {'backbone': [(model['backbone_url'] + NAME % (i + 1), NAME % (i + 1), s) for i, s in enumerate(SIZES)],
         'engram': [(HF + ENGRAM, ENGRAM, 110595290816)],
         'draft': [(DRAFT, 'dspark.gguf', 7976914240)]}[what]
with open(out, 'w') as f:
    json.dump([{'url': u, 'dest': os.path.join(dest, n), 'dest_size': s, 'src_off': 0, 'dest_off': 0, 'size': s} for u, n, s in files], f)
print('%s: %.2f GB' % (what, sum(s for _, _, s in files) / 1e9))
