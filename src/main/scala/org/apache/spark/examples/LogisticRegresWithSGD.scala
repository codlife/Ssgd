package org.apache.spark.examples


import org.apache.spark.mllib.classification.{LogisticRegressionWithAdagrad, LogisticRegressionWithSGDMomentum, LogisticRegressionWithSGDSVRG2, LogisticRegressionWithSgd}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by wjf on 16-11-30.
  */
object LogisticRegresWithSGD {

  def main(args: Array[String]): Unit = {


    val conf = new SparkConf().setAppName("ssgd")

    val sc = new SparkContext(conf)
    val iteration = args(args.length -7).toInt
    val path = args(args.length -6)
    val number = args(args.length -5).toInt
    val algorithm = args(args.length -4)
    val sampleForSVRG = args(args.length -3)
    val sampleFraction = args(args.length -2)
    val convergeTo = args(args.length -1)

    // $example on$
    // Load training data in LIBSVM format.
    //    val data = MLUtils.loadLibSVMFile(sc, "/usr/data/ccf_data/regression/trainR10_libsvm.csv")
    //        val data = MLUtils.loadLibSVMFile(sc,"hdfs://133.133.10.1:9000/user/lijie/data/wjf/covtype.txt")
//    val data = MLUtils.loadLibSVMFile(sc,"/usr/data/ccf_data/regression/trainR10_libsvm.csv")

//    val data = MLUtils.loadLibSVMFile(sc,"/usr/local/intellij/splash-master/examples/data/covtype.txt").repartition(4)
    val data = MLUtils.loadLibSVMFile(sc,path).repartition(number)

    val splits = data.map(x =>
      if(x.label == 1) {
        LabeledPoint(x.label, x.features)
      } else {
        LabeledPoint(0,x.features)
      }).randomSplit(Array(0.7, 0.3), seed = 11L)
    val training = splits(0).cache()
    val test = splits(1)

    val start = System.nanoTime()
   if(algorithm == "momentum") {
     println("momentum is running")
     val sgd = new LogisticRegressionWithSGDMomentum(1.0, iteration, 1, sampleFraction.toDouble,convergeTo.toDouble)
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

   } else if(algorithm == "sgd"){


     println("sgd is running")
     val sgd = new LogisticRegressionWithSgd(1.0, iteration, 1, sampleFraction.toDouble,convergeTo.toDouble)
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
   } else if(algorithm == "svrg") {



     println("svrg is running")
     val sgd = new LogisticRegressionWithSGDSVRG2(1.0, iteration, 1, sampleFraction.toDouble,sampleForSVRG.toDouble,convergeTo.toDouble)
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
   } else if(algorithm == "adagrad") {



     println("adagrad is running")
     val sgd = new LogisticRegressionWithAdagrad(1.0, iteration, 1, sampleFraction.toDouble,convergeTo.toDouble)
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
   }

    sc.stop()
  }
}