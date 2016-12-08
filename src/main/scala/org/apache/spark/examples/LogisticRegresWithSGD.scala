package org.apache.spark.examples

import org.apache.spark.mllib.classification.{LogisticRegressionWithSGD, LogisticRegressionWithSGDMomentum, LogisticRegressionWithSGDSVRG}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.optimization.{LogisticGradient, SquaredL2Updater}
import org.apache.spark.mllib.optimization.SGD.GradientDescent
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by wjf on 16-11-30.
  */
object LogisticRegresWithSGD {

  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("LogisticRegressionWithLBFGSExample").setMaster("local[*]")
    val sc = new SparkContext(conf)

    // $example on$
    // Load training data in LIBSVM format.
    //    val data = MLUtils.loadLibSVMFile(sc, "/usr/data/ccf_data/regression/trainR10_libsvm.csv")
    //        val data = MLUtils.loadLibSVMFile(sc,"hdfs://133.133.10.1:9000/user/lijie/data/wjf/covtype.txt")
    val data = MLUtils.loadLibSVMFile(sc,"/usr/data/ccf_data/regression/trainR10_libsvm.csv")
    //    val testdata = MLUtils.loadLibSVMFile(sc, "/usr/data/regression/testR10_libsvm.csv")


    // Split data into training (60%) and test (40%).
    val splits = data.randomSplit(Array(0.6, 0.4), seed = 11L)
    val training = splits(0).cache()
    val test = splits(1)
    //
    //    // Run training algorithm to build the model
    //
    //    // for verify
    val start = System.nanoTime()
//    val sgd = new LogisticRegressionWithSGD(1.0, 50, 0.01, 1.0)
    val sgd = new LogisticRegressionWithSGDMomentum(1.0, 50, 0.01, 1.0)

//    val sgd = new LogisticRegressionWithSGDSVRG(1.0, 50, 0.01, 1.0)


    val model = sgd.run(training)
    model.clearThreshold()
    val predictionAndLabels = test.map { case LabeledPoint(label, features) =>
      val prediction = model.predict(features)
      (label, prediction)
    }

    val metric = new BinaryClassificationMetrics(predictionAndLabels)
    val auROC = metric.areaUnderROC()
    println("AREA under ROC = " + auROC)


    println(data.count())
    val end = System.nanoTime()

    println("this is time" + (end - start) / 1000000)



    sc.stop()
  }
}