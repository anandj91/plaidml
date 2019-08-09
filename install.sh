set -e
bazel build --config linux_x86_64 plaidml:wheel plaidml/keras:wheel
pip install -U bazel-bin/plaidml/*whl bazel-bin/plaidml/keras/*whl
