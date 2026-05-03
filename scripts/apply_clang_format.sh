#!/usr/bin/env bash
set -euo pipefail

find $PWD/include/ $PWD/src/ $PWD/tests/ $PWD/examples  $PWD/benchmark \
  -type f \
  -regex '.*\.\(cu\|cuh\|cpp\|cxx\|h\)$' \
  -print0 |
xargs -0 clang-format -i

echo "All files clang formatted"