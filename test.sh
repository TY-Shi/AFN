gpuid=3

CUDA_VISIBLE_DEVICES=${gpuid} python -u test.py --config config/AFN.yaml
