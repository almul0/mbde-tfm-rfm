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
from pyspark.ml.feature import VectorAssembler

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

def compute_rfm(spark, input, quartile_output, r_percentile=None,f_percentile=None,m_percentile=None):
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

    logger.info('Calculating upper boundaries')
    r_trim_cond = f_trim_cond = m_trim_cond = F.lit(True)
    r_out_cond = f_out_cond = m_out_cond = F.lit(False)

    if r_percentile:
        r_up_boundary = rfm_table.approxQuantile("recency", [r_percentile], 0)
        r_trim_cond = (F.col("recency") < r_up_boundary[0])
        r_out_cond = (F.col("recency") >= r_up_boundary[0])

    if f_percentile:
        f_up_boundary = rfm_table.approxQuantile("frequency", [f_percentile], 0)
        f_trim_cond = (F.col("frequency") < f_up_boundary[0])
        f_out_cond = (F.col("frequency") >= f_up_boundary[0])

    if f_percentile:
        m_up_boundary = rfm_table.approxQuantile("monetary", [m_percentile], 0)
        m_trim_cond = (F.col("monetary") < m_up_boundary[0])
        m_out_cond = (F.col("monetary") >= m_up_boundary[0])


    logger.info('Trimming data to compute quartiles')
    rfm_trimmed = rfm_table
    rfm_trimmed = rfm_trimmed.filter(r_trim_cond & f_trim_cond & m_trim_cond)
    rfm_outliers = rfm_table
    rfm_outliers = rfm_outliers.filter(r_out_cond | f_out_cond | m_out_cond)

    total_count = rfm_table.count()
    trimmed_count = rfm_trimmed.count()
    outliers_count = rfm_outliers.count()
    logger.info(f"Total = {total_count}")
    logger.info(f"Trimmed = {trimmed_count}")
    logger.info(f"Outliers = {outliers_count}")

    #create quartiles for each metric
    logger.info('Computing quartiles')
    r_quartile = rfm_trimmed.approxQuantile("recency", [0.25, 0.5, 0.75], 0)
    f_quartile = rfm_trimmed.approxQuantile("frequency", [0.25, 0.5, 0.75], 0)
    m_quartile = rfm_trimmed.approxQuantile("monetary", [0.25, 0.5, 0.75], 0)

    spark.createDataFrame(r_quartile, FloatType()).write.mode('overwrite').csv(quartile_output+"/r_quartile")
    spark.createDataFrame(f_quartile, FloatType()).write.mode('overwrite').csv(quartile_output+"/f_quartile")
    spark.createDataFrame(m_quartile, FloatType()).write.mode('overwrite').csv(quartile_output+"/m_quartile")

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

def extract(row):
    return tuple(row.scaled_features.toArray().tolist())

def calculate_initial_centroids(spark, features):
    rm_features = features.select('scaled_features').rdd.map(extract).toDF()
    rows_count = rm_features.count()
    chunk_size=np.ceil(rows_count/N_CLUSTERS)
    bc_chunk_size = spark.sparkContext.broadcast(chunk_size)

    r_vector = rm_features.select('_1').rdd.sortBy(lambda x: x[0]).zipWithIndex().map(lambda x: (x[0][0],int(x[1]/bc_chunk_size.value))).toDF()
    f_vector = rm_features.select('_2').rdd.sortBy(lambda x: x[0]).zipWithIndex().map(lambda x: (x[0][0],int(x[1]/bc_chunk_size.value))).toDF()
    m_vector = rm_features.select('_3').rdd.sortBy(lambda x: x[0]).zipWithIndex().map(lambda x: (x[0][0],int(x[1]/bc_chunk_size.value))).toDF()

    r_median = r_vector.groupby('_2').agg(F.expr('percentile_approx(_1, 0.5)').alias('r_median'))
    f_median = f_vector.groupby('_2').agg(F.expr('percentile_approx(_1, 0.5)').alias('f_median'))
    m_median = m_vector.groupby('_2').agg(F.expr('percentile_approx(_1, 0.5)').alias('m_median'))

    rfm_medians = r_median.join(f_median,'_2').join(m_median,'_2').sort('_2').select(['r_median','f_median','m_median'])
    initial_centroids = rfm_medians.rdd.map(lambda x: np.array([x[0],x[1],x[2]])).collect()
    return initial_centroids

def train_kmeans(train_features, initial_model):
    model = KMeans.train(train_features, N_CLUSTERS ,initialModel=initial_model)
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input',
        help="S3 location for transaction files")
    parser.add_argument('--model-output', help="K-Means model output")
    parser.add_argument('--quartile-output', help="K-Means model output")
    parser.add_argument('--single-files', action="store_true", help="Single file output flag")
    parser.add_argument('--r-percentile',type=float)
    parser.add_argument('--f-percentile',type=float)
    parser.add_argument('--m-percentile',type=float)
    parser.add_argument('--prediction-output', help="S3 from output prediction")
    args = parser.parse_args()

    with SparkSession.builder.appName("Train segments").getOrCreate() as spark:
        spark.sparkContext._jsc.hadoopConfiguration().set("fs.s3.canned.acl","BucketOwnerFullControl")

        rfm_table = compute_rfm(spark, args.input, args.quartile_output,
                        r_percentile= args.r_percentile,
                        f_percentile= args.f_percentile,
                        m_percentile=  args.m_percentile)

        # Standarize features
        logger.info("Standarize features")
        features = standarize_features(rfm_table)

        # Centroid initializacion
        initial_centroids = calculate_initial_centroids(spark, features)
        logger.info(f"Initial centroids: {initial_centroids}")

        initial_model = KMeansModel(initial_centroids)

        train_features = features.select('scaled_features').rdd.map(lambda x: x[0].toArray()).cache()
        logger.info("Training K-Means")
        model = train_kmeans(train_features, initial_model)

        logger.info("Saving model....")
        model.save(spark, args.model_output)

        logger.info("Prediction....")
        labels = features.select(['customerid','scaled_features']).rdd.map(lambda x: (x[0], model.predict(x[1].toArray()))).toDF(['customerid','prediction'])

        prediction = features.join(labels,'customerid')

        prediction_map = dict((v,k) for k,v in PREDICTION_MAP.items())
        logger.info("Segment mapping")
        map_func = F.udf(lambda row : prediction_map.get(row,row))
        prediction = prediction.withColumn("segmento",map_func(F.col('prediction')))
        prediction = prediction.withColumn("prediction_label",F.concat(F.lit('('),F.col('prediction'),F.lit(') '),F.col('segmento')))

        logger.info("Saving")
        prediction.write.mode('overwrite').format('json').save(args.prediction_output + "/all")
        for k in PREDICTION_MAP:
            prediction.filter(F.col('segmento') == k).write.mode('overwrite').format('json').save(args.prediction_output + "/" + k)


