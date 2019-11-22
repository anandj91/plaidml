#! /bin/bash -e
bazel build --jobs 20 --config linux_x86_64 plaidml:wheel plaidml/keras:wheel
pip install -U bazel-bin/plaidml/*whl bazel-bin/plaidml/keras/*whl
echo -e "\a"
