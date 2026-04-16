#!/usr/bin/env bash
set -euo pipefail

find $PWD/include/ $PWD/src/ $PWD/tests/ \
  -type f \
  -regex '.*\.\(cu\|cuh\|cpp\|cxx\|h\)$' \
  -print0 |
xargs -0 clang-format --dry-run --Werror

echo "clang-format check passed."