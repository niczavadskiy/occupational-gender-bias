# Prob INLP on Qwen3.5-4B-Base — peak + pre-peak
#
# Targets: gender_prob / slot_prob only (Ridge). Sample:
#   experiments/qwen35-4b-base/steering/samples/h1_stagea_sample_v1.json
#
# ## Peak (classical single-layer Stage A)
#
# Layers: gender 23–25 · slot 29–31 (choice peak belts).
#
#   export HF_TOKEN=hf_xxx
#   cd /workspace/occupational-gender-bias && git pull
#   bash experiments/qwen35-4b-base/steering/scripts/vast_inlp_gender_prob_stagea_and_pack.sh
#   bash experiments/qwen35-4b-base/steering/scripts/vast_inlp_slot_prob_stagea_and_pack.sh
#
# Smoke: SMOKE=1 bash …/vast_inlp_gender_prob_stagea_and_pack.sh
#
# ## Pre-peak (sequential ascent pool)
#
#   AXIS=both bash experiments/qwen35-4b-base/steering/scripts/vast_sequential_pre_peak_4b_and_pack.sh
#
# ## All four runs
#
#   bash experiments/qwen35-4b-base/steering/scripts/vast_prob_peak_and_prepeak_4b_and_pack.sh
#   MODE=peak|prepeak|both
#
# ## Outputs (/workspace)
#
# - inlp_gender_prob_inlp_gender_prob_4b_a_v1.tar.gz  (ranking.csv + npz)
# - inlp_slot_prob_inlp_slot_prob_4b_a_v1.tar.gz
# - sequential_pre_peak_prepeak_4b_v1.tar.gz  (summary.json + prepeak npz)
#
# Stub auc_curve in pre-peak yamls mirrors choice layer_scan shape; replace
# with real gender_prob / slot_prob layer_scan via config layer_scan.path.
