#!/bin/bash
# MIMIC-Eye 定向并行下载:每个 patient 只取 ED/ + CXR-JPG/ + CXR-DICOM/
cd ~/mimic_eye || exit 1
B="https://physionet.org/files/mimic-eye-multimodal-datasets/1.0.0/mimic-eye"
grab() {
  p="$1"
  for d in ED CXR-JPG CXR-DICOM; do
    wget -q -r -N -np -nH --cut-dirs=3 --tries=2 --timeout=30 "$B/$p/$d/" 2>/dev/null
  done
}
export -f grab; export B
xargs -a patients.txt -I{} -P 16 bash -c 'grab "$@"' _ {}
echo "MIMIC_EYE_DL_DONE"
du -sh ~/mimic_eye
find ~/mimic_eye -name "*.jpg" | wc -l
