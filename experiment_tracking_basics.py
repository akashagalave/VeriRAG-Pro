import mlflow
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent

mlflow.set_tracking_uri("https://dagshub.com/akashagalaveaaa1/VeriRAG-Pro.mlflow")

mlflow.set_experiment("verirag_pro_experiment")


with mlflow.start_run(run_name="test_run") as run:
    ##params
    mlflow.log_param("chunk_size",300)
    mlflow.log_param("chunk_overlap",30)
    mlflow.log_param("output_dims",1024)

    mlflow.log_metric("recall",0.89)
    mlflow.log_metric("faithfullness",0.91)


    ## log artifact
    mlflow.log_artifact(local_path=(ROOT_DIR / "backend" /"rag_graph.py"),
                       artifact_path="code" )

    ## log dataset
    mlflow.log_artifact(local_path=(ROOT_DIR /"goldens.json"))
