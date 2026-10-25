#!/bin/bash
cd ~/digital_typhoon
cat WP.tar.gzaa WP.tar.gzab | tar xzf -
echo "EXTRACT_DONE"
echo "=== 结构 ==="
ls WP/
echo "=== 序列数 ==="; ls WP/image/ | wc -l
echo "=== h5 总数 ==="; find WP/image -name "*.h5" | wc -l
echo "=== metadata ==="; ls WP/metadata* 2>/dev/null | head; find WP -maxdepth 2 -name "*.csv" | head
du -sh WP
