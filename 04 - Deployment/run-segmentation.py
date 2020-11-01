import argparse
import logging

from pyspark.sql import SparkSession

import numpy as np
from math import sqrt
import pyspark.sql.functions as F
from pyspark.sql.types import *

from pyspark.mllib.clustering import KMeans,KMeansModel
from pyspark.mllib.linalg import Vectors
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import StandardScaler
# from pyspark.sql import SQLContext
# from pyspark.sql.functions import col, when, lit, expr, countDistinct, max, min, sum, concat
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import StandardScaler

N_CLUSTERS=6

PREDICTION_MAP = {
    'best': 0,
    'ocassional': 1,
    'loyal': 2,
    'promising': 3,
    'risk': 4,
    'lost' :5
}

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
spark = None

def compute_rfm(spark, input, quartile_input):
    logger.info("Loading data from {}".format(input))
    tickets = spark.read.format("json").load(input)

    # Drop duplicate rows
    tickets = tickets.drop_duplicates()
    tickets = tickets.withColumn("date", F.to_date("datekey", "yyyy-MM-dd"))
    logger.info('Grouping tickets')
    tickets = tickets.groupby(['customerid','storeid','cardtype','date','ticketid','productid']).agg(
        F.sum('extendedamount').alias('extendedamount'),
        F.sum('originalamount').alias('originalamount'),
        F.sum('totaldiscount').alias('totaldiscount'),
        F.sum('quantity').alias('quantity')
    )
    logger.info('Calculating RFM')
    max_df = tickets.select(F.max("date")).collect()
    max_val = max_df[0][0]
    tickets = tickets.withColumn('max_date',F.lit(max_val))
    tickets = tickets.withColumn("recencydays", F.expr("datediff(max_date, date)"))
    rfm_table = tickets.groupBy("customerid")\
                        .agg(F.min("recencydays").alias("recency"), \
                            F.countDistinct("ticketid").alias("frequency"), \
                            F.sum("extendedamount").alias("monetary"))

    r_quartile = sorted([row[0] for row in spark.read.csv(quartile_input+"/r_quartile").collect()])
    f_quartile = sorted([row[0] for row in spark.read.csv(quartile_input+"/f_quartile").collect()])
    m_quartile = sorted([row[0] for row in spark.read.csv(quartile_input+"/m_quartile").collect()])

    #assing score from 1 to 4 dependending on quantile => 1 best mark - 4 worst one
    logger.info('Assigning scores')
    rfm_table = rfm_table.withColumn("r_quartile",F.when(F.col("recency") > r_quartile[2] , 4).\
                                                            when(F.col("recency") > r_quartile[1] , 3).\
                                                            when(F.col("recency") > r_quartile[0] , 2).\
                                                            otherwise(1))

    rfm_table = rfm_table.withColumn("f_quartile",F.when(F.col("frequency") > f_quartile[2] , 1).\
                                                            when(F.col("frequency") > f_quartile[1] , 2).\
                                                            when(F.col("frequency") > f_quartile[0] , 3).\
                                                            otherwise(4))

    rfm_table = rfm_table.withColumn("m_quartile",F.when(F.col("monetary") > m_quartile[2] , 1).\
                                                            when(F.col("monetary") > m_quartile[1] , 2).\
                                                            when(F.col("monetary") > m_quartile[0] , 3).\
                                                            otherwise(4))

    rfm_table = rfm_table.withColumn("rfm_score", F.concat(F.col("r_quartile"), F.col("f_quartile"), F.col("m_quartile")))

    return rfm_table

def standarize_features(rfm_table):
    assembler = VectorAssembler(inputCols=['r_quartile','f_quartile','m_quartile'],\
                            outputCol='features',handleInvalid = 'skip')
    rfm_table = assembler.transform(rfm_table)

    standardizer = StandardScaler(withMean=True, withStd=True).setInputCol("features").setOutputCol("scaled_features")
    std_model = standardizer.fit(rfm_table)
    features = std_model.transform(rfm_table)
    return features

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input',
        help="S3 location for transaction files")
    parser.add_argument('--model-input', help="K-Means model input")
    parser.add_argument('--quartile-input', help="Quartile input")
    parser.add_argument('--single-files', action="store_true", help="Single file output flag")
    parser.add_argument(
        '--prediction-output', help="S3 from output prediction")
    args = parser.parse_args()

    with SparkSession.builder.appName("Train segments").getOrCreate() as spark:
        spark.sparkContext._jsc.hadoopConfiguration().set("fs.s3.canned.acl","BucketOwnerFullControl")

        rfm_table = compute_rfm(spark, args.input, args.quartile_input)

        # Standarize features
        logger.info("Standarize features")
        features = standarize_features(rfm_table)

        train_features = features.select('scaled_features').rdd.map(lambda x: x[0].toArray()).cache()
        logger.info("Training K-Means")

        model = KMeansModel.load(spark.sparkContext, args.model_input)

        logger.info("Prediction....")
        labels = features.select(['customerid','scaled_features']).rdd.map(lambda x: (x[0], model.predict(x[1].toArray()))).toDF(['customerid','prediction'])

        prediction = features.join(labels,'customerid')

        prediction_map = dict((v,k) for k,v in PREDICTION_MAP.items())
        logger.info("Segment mapping")
        map_func = F.udf(lambda row : prediction_map.get(row,row))
        prediction = prediction.withColumn("segmento",map_func(F.col('prediction')))
        prediction = prediction.withColumn("prediction_label",F.concat(F.lit('('),F.col('prediction'),F.lit(') '),F.col('segmento')))

        logger.info(prediction.take(5))

        logger.info("Saving")
        prediction.write.mode('overwrite').format('json').save(args.prediction_output + "/all")
        for k in PREDICTION_MAP:
            prediction.filter(F.col('segmento') == k).write.mode('overwrite').format('json').save(args.prediction_output + "/" + k)