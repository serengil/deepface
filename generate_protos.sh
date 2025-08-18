#!/bin/bash
set -e

# Source and destination for .proto files
PROTO_SRC_DIR="deepface/api/proto"
PROTO_DST_DIR="deepface/api/proto"

# Locate grpc_tools _proto directory for well-known Google protos
GRPC_TOOLS_PROTO_DIR=$(python -c "import grpc_tools; import os; print(os.path.join(os.path.dirname(grpc_tools.__file__), '_proto'))")

echo "Generating protos..."
python -m grpc_tools.protoc \
  --proto_path="$PROTO_SRC_DIR" \
  --proto_path="$GRPC_TOOLS_PROTO_DIR" \
  --python_out="$PROTO_DST_DIR" \
  --grpc_python_out="$PROTO_DST_DIR" \
  --pyi_out="$PROTO_DST_DIR" \
  "$PROTO_SRC_DIR"/*.proto

echo "Fixing imports to use relative paths..."
# Fix `import X_pb2` → `from . import X_pb2`
# We want to check if this is ran inside a MacOS environment or Linux
# as sed -i behaves differently on each platform.
case "$OSTYPE" in
  darwin*|bsd*)
    sed_no_backup=( -i '' )
    ;; 
  *)
    sed_no_backup=( -i )
    ;;
esac

find "$PROTO_DST_DIR" -name "*.py" -exec sed "${sed_no_backup[@]}" -E 's/^import (.+_pb2)/from . import \1/' {} \;

echo "Adding 'import grpc.experimental' after 'import grpc' in *_pb2_grpc.py files..."
find "$PROTO_DST_DIR" -name "*_pb2_grpc.py" -exec sed "${sed_no_backup[@]}" -E '/^import grpc$/a\
import grpc.experimental
' {} \;

echo "Done."
