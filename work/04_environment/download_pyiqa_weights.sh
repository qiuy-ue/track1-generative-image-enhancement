#!/bin/bash
cd /root/.cache/torch/hub/pyiqa || exit 1
for f in brisque_svm_weights.pth cfanet_nr_koniq_res50-9a73138b.pth P2P_RoIPoolModel-fit.10.bs.120-ca69882e.pth; do
  if [ ! -s "$f" ]; then
    curl -L --fail --retry 3 -o "$f" "https://hf-mirror.com/chaofengc/IQA-PyTorch-Weights/resolve/main/$f"
  fi
done
ls -lh brisque_svm_weights.pth cfanet_nr_koniq_res50-9a73138b.pth P2P_RoIPoolModel-fit.10.bs.120-ca69882e.pth
