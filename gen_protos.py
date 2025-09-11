# uv pip install betterproto2_compiler grpcio-tools
import sys
from pathlib import Path
from grpc_tools import protoc
import os
import importlib.resources
import shutil

def gen(protos, outdir):
    with importlib.resources.path("grpc_tools", "_proto") as proto_include:
        protos_include_cp = protos / "google/protobuf"
        protos_include_cp.mkdir(exist_ok=True, parents=True)
        for p in proto_include.glob("**/*.proto"):
            shutil.copy(p, protos_include_cp / p.name)
        files = [str(f.relative_to(protos)) for f in protos.glob("**/*.proto")]
        outdir.mkdir(exist_ok=True)
        cwd = os.getcwd()  # Save current directory
        try:
            os.chdir(protos)  # Change to output directory
            res = protoc.main(
                [
                    "",
                    f"--python_betterproto2_out={outdir}",
                    "--python_betterproto2_opt=google_protobuf_descriptors",
                    *files,
                ]
            )  # Use '.' as output
        finally:
            os.chdir(cwd)  # Restore original directory
        if res != 0:
            sys.exit(res)

waymo_proto_dir = Path(__file__).parent/'src/trajdata/dataset_specific/waymo/waymo_proto/protos/'
gen(
    waymo_proto_dir,
    waymo_proto_dir.parent/('./generated_stubs')
)

trajdata_dir = Path(__file__).parent/'src/trajdata/proto'
gen(
    trajdata_dir,
    trajdata_dir/'generated_stubs'
)
