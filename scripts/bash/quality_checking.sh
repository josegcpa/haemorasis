NET_PATH=scripts/python/attention_classifier.py
SLIDE_PATH=$1
CKPT_PATH=$2
PLACEHOLDER_STR=n
PLACEHOLDER_FLOAT=0.01
PLACEHOLDER_INT=1

python3 $NET_PATH\
  $CKPT_PATH\
  $SLIDE_PATH\
  0.0001\
  100000\
  32\
  predict\
  N\
  False\
  2
