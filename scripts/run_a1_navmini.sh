#!/usr/bin/env bash
# [MAC] A1: ConstantVelocity tren navmini (10 log chuan cua NAVSIM), frame_interval=1.
# Chay: metric caching -> non_reactive -> reactive. Dung ngay neu mot buoc loi
# (ma thoat != 0 hoac khong co CSV) va ghi ly do vao STATUS_RUN.txt.
# Cache rieng: $NAVSIM_EXP_ROOT/metric_cache_navmini_fi1 (khong de len cache mini cu).
set -u
W=$HOME/navsim_workspace
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0" NUPLAN_MAPS_ROOT=$W/dataset/maps \
  NAVSIM_EXP_ROOT=$W/exp NAVSIM_DEVKIT_ROOT=$W/navsim OPENSCENE_DATA_ROOT=$W/dataset
PY=$HOME/miniforge3/envs/navsim_v2/bin/python
RUN=$W/research/logs/A1_navmini_$(date +%Y%m%d-%H%M); mkdir -p "$RUN"
OUT=$W/research/results/A1_cv_navmini; mkdir -p "$OUT"
C=$NAVSIM_EXP_ROOT/metric_cache_navmini_fi1
SF="train_test_split=navmini train_test_split.scene_filter.frame_interval=1"
echo "env=navsim_v2 devkit=$NAVSIM_DEVKIT_ROOT navsim_commit=$(git -C $W/navsim rev-parse --short HEAD) start=$(date)" | tee "$RUN/STATUS_RUN.txt"
cd $W/navsim || exit 1
fail(){ echo "FAILED at $1 $(date)" | tee -a "$RUN/STATUS_RUN.txt"; exit 1; }

$PY navsim/planning/script/run_metric_caching.py $SF metric_cache_path=$C worker=sequential > "$RUN/cache.log" 2>&1 \
  || fail cache
[ -f "$C/metadata/$(ls $C/metadata | grep -m1 csv)" ] || fail cache_csv
echo "cache done $(date) n_logs=$(ls -d $C/20* | wc -l)" | tee -a "$RUN/STATUS_RUN.txt"

for t in non_reactive reactive; do
  $PY navsim/planning/script/run_pdm_score_one_stage.py $SF metric_cache_path=$C \
    agent=constant_velocity_agent experiment_name=A1_cv_navmini_$t traffic_agents=$t worker=sequential \
    > "$RUN/score_$t.log" 2>&1 || fail score_$t
  CSV=$(ls -t $NAVSIM_EXP_ROOT/A1_cv_navmini_$t/*/*.csv 2>/dev/null | head -1)
  [ -n "$CSV" ] || fail csv_$t
  cp "$CSV" "$OUT/cv_navmini_$t.csv"
  echo "$t done $(date) csv=$OUT/cv_navmini_$t.csv" | tee -a "$RUN/STATUS_RUN.txt"
  grep -E "successful|failed scenarios|Final average" "$RUN/score_$t.log" | tee -a "$RUN/STATUS_RUN.txt"
done
echo "ALL DONE $(date)" | tee -a "$RUN/STATUS_RUN.txt"
