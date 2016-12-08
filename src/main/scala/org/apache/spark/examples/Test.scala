package org.apache.spark.examples

import org.apache.spark.sql.SparkSession

/**
  * Created by wjf on 16-12-7.
  */
object Test {
  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder().master("local").appName("test sgd").getOrCreate()

    val rdd1 = spark.sparkContext.parallelize(Seq(1,2,3,4),2)
    rdd1.mapPartitions {
      partition => {
        val x = partition.next()
        Iterator(x)
      }
    }.collect().foreach(println)
  }

}
