#!/bin/bash

# 可选的第三个 flag（可传可不传）
extra_flag="$1"

# 要跑的矩阵名（不带路径后缀）
matrices=( "onetone1" "onetone2" "bcircuit" "G2_circuit" "hcircuit" "parabolic_fem" )

# 默认三种模式：无 flag、CA、LA
opts=("LA_OPT") #"LA" "" "CA" 

size=()
# 如果用户传了 extra_flag，就追加到 size 数组里
if [ -n "$extra_flag" ]; then
  size+=( "$extra_flag" )
fi

for m in "${matrices[@]}"; do
  file="data/$m/$m.mtx"
  for opt in "${opts[@]}"; do
    for i in {1..20}; do
      if [ -z "$opt" ]; then
        echo "===== $m DAG Run #$i ====="
        ./run_sparse "$file"
      else
        echo "===== $m $opt Run #$i ====="
        if [ -z "$size" ]; then
            ./run_sparse "$file" "$opt"
        else
            ./run_sparse "$file" "$opt" "$size"
        fi
      fi
    done
  done
done