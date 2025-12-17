
#!/usr/bin/env bash
set -eo pipefail

################################################################################
# 1) Make sure conda’s shell functions are available in this script:
#    Option A: using the "conda shell.bash hook":
eval "$(conda shell.bash hook)"


ROOT_DIR="$1"

sh 6_align.sh "$ROOT_DIR" gv
sh re_glue_sqs.sh "$ROOT_DIR" gv



