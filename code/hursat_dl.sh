#!/bin/bash
B="https://www.ncei.noaa.gov/data/hurricane-satellite-hursat-b1/archive/v06"
cd ~/hursat || exit 1
# 收集 2000-2015 的所有风暴 tar.gz 链接
> urls.txt
for y in $(seq 2000 2015); do
  curl -s --max-time 60 "$B/$y/" | grep -oE '<a href="[^"]+\.tar\.gz">' | sed 's/<a href="//;s/">//' \
    | while read f; do echo "$B/$y/$f"; done >> urls.txt
done
echo "总风暴数: $(wc -l < urls.txt)"
mkdir -p raw
cd raw
xargs -a ../urls.txt -P 8 -I{} wget -q -c -nc {} 2>/dev/null
echo "HURSAT_DL_DONE"
ls *.tar.gz | wc -l
du -sh .
