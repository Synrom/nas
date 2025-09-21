# Computer Vision Lab about NAS

## Setup

The dependencies can be installed using:
```bash
pip install -r requirements.txt
```

## Search

To perform search as done in the lab:
```bash
python train_search_ppc.py --no-vis-eigenvalues --stages  ppc_stage.json  --unrolled --fair --runid searchfair
```

## Evaluation

To train the resulting architecture on CIFAR-10:
```bash
python train_eval.py  --runid eval_searchfair --cutout --auxiliary --genotype log/searchfair/genotype_2_39.json --batch_size 128
```

To train on CIFAR-100:
```bash
python train_eval.py  --runid eval_searchfair --cutout --auxiliary --genotype log/searchfair/genotype_2_39.json --batch_size 128 --cifar100
```