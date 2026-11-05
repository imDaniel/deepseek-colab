"""Check that every linked shard has its published byte size. usage: python3 -I checksizes.py MODEL_DIR"""
import json, os, sys
RUNG = 'IQ2_XXS'
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models.json')) as f:
    models = json.load(f)
SIZES = dict(enumerate(models[os.environ.get('MODEL_VARIANT', 'original')]['sizes'] + [110595290816], 1))
bad = 0
for i, n in SIZES.items():
    p = os.path.join(sys.argv[1], 'DeepSeek-V4.1-Flash-1017-GSQ-RCO-%s-%05d-of-00006.gguf' % (RUNG, i))
    s = os.path.getsize(p) if os.path.exists(p) else -1
    print(i, os.path.realpath(p), s, 'ok' if s == n else 'BAD')
    bad += s != n
sys.exit(1 if bad else 0)
