#!/bin/bash
cd ~/digital_typhoon
B=https://huggingface.co/datasets/torchgeo/digital_typhoon/resolve/main
for f in WP.tar.gzaa WP.tar.gzab; do
  echo "=== downloading $f ==="
  wget -c -q --show-progress -O $f "$B/$f" 2>&1 | tail -2
done
echo "=== sizes ==="; ls -la WP.tar.gz*
echo "DT_DOWNLOAD_DONE"
