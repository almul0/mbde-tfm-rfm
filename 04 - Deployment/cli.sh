aws emr create-default-roles

aws emr create-cluster \
    --release-label emr-5.30.0 \
    --instance-type m5.xlarge --instance-count 2 --applications Name=Spark Name=Hadoop \
    --use-default-roles \
    --scale-down-behavior TERMINATE_AT_TASK_COMPLETION \
    --region us-east-1

CLUSTER_ID="j-XXXXXXXXXXXX"

aws s3 rm s3://mbde-tfm-grupo3/kmeans_model --recursive
aws s3 rm s3://mbde-tfm-grupo3/train_output --recursive
aws s3 rm s3://mbde-tfm-grupo3/quartiles --recursive
aws emr add-steps --cluster-id ${CLUSTER_ID} --steps file://./train-steps.json --region us-east-1