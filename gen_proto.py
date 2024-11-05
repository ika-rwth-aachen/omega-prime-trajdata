
# uv pip install betterproto2-compiler grpcio-tools

from pathlib import Path
from grpc_tools import protoc
import os

waymo_proto_dir = Path(__file__).parent/'src/trajdata/dataset_specific/waymo/waymo_proto/protos'
files = [str(f.relative_to(waymo_proto_dir)) for f in waymo_proto_dir.glob("**/*.proto")]
outdir = waymo_proto_dir/('../generated_stubs')
outdir.mkdir(exist_ok=True)
cwd = os.getcwd()  # Save current directory
try:
    os.chdir(waymo_proto_dir)  # Change to output directory
    result = protoc.main(['', f'--python_betterproto2_out={outdir}', *files])  # Use '.' as output
finally:
    os.chdir(cwd)  # Restore original directory
    
    
trajdata_dir = Path(__file__).parent/'src/trajdata/proto'
files = [str(f.relative_to(trajdata_dir)) for f in trajdata_dir.glob("**/*.proto")]
outdir = trajdata_dir/('./generated_stubs')
outdir.mkdir(exist_ok=True)
cwd = os.getcwd()  # Save current directory
try:
    os.chdir(trajdata_dir)  # Change to output directory
    result = protoc.main(['', f'--python_betterproto2_out={outdir}', *files])  # Use '.' as output
finally:
    os.chdir(cwd)  # Restore original directory