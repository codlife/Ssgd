package org.apache.spark.examples


import org.apache.spark.mllib.classification.{LogisticRegressionWithAdagram, LogisticRegressionWithSGD, LogisticRegressionWithSGDMomentum, LogisticRegressionWithSGDSVRG2}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by wjf on 16-11-30.
  */
object LogisticRegresWithSGD {

  def main(args: Array[String]): Unit = {


    val conf = new SparkConf().setAppName("LogisticRegressionWithLBFGSExample")

    val sc = new SparkContext(conf)
    val iteration = args(args.length -6).toInt
    val path = args(args.length -5)
    val number = args(args.length -4).toInt
    val algorithm = args(args.length -3)
    val sampleForSVRG = args(args.length -2)
    val sampleFraction = args(args.length -1)

    // $example on$
    // Load training data in LIBSVM format.
    //    val data = MLUtils.loadLibSVMFile(sc, "/usr/data/ccf_data/regression/trainR10_libsvm.csv")
    //        val data = MLUtils.loadLibSVMFile(sc,"hdfs://133.133.10.1:9000/user/lijie/data/wjf/covtype.txt")
//    val data = MLUtils.loadLibSVMFile(sc,"/usr/data/ccf_data/regression/trainR10_libsvm.csv")

//    val data = MLUtils.loadLibSVMFile(sc,"/usr/local/intellij/splash-master/examples/data/covtype.txt").repartition(4)
    val data = MLUtils.loadLibSVMFile(sc,path).repartition(number)


    //    val testdata = MLUtils.loadLibSVMFile(sc, "/usr/data/regression/testR10_libsvm.csv")


    // Split data into training (60%) and test (40%).
    val splits = data.map(x =>
      if(x.label == 1) {
        LabeledPoint(x.label, x.features)
      } else {
        LabeledPoint(0,x.features)
      }).randomSplit(Array(0.7, 0.3), seed = 11L)
    val training = splits(0).cache()
    val test = splits(1)
    //
    //    // Run training algorithm to build the model
    //
    //    // for verify
    val start = System.nanoTime()

//    val sgd = new LogisticRegressionWithSGD(1.0, 10, 1, 1.0)

   if(algorithm == "momentum") {
     println("momentum is running")
     val sgd = new LogisticRegressionWithSGDMomentum(1.0, iteration, 1, sampleFraction.toDouble)
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
     val sgd = new LogisticRegressionWithSGD(1.0, iteration, 1, sampleFraction.toDouble)
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

//    val sgd = new LogisticRegressionWithSGD(1.0, 50, 0.01, 1.0)
    //   val sgd = new LogisticRegressionWithSGDMomentum(1.0, 50, 0.01, 1.0)



     println("this is time" + (end - start) / 1000000)
   } else if(algorithm == "svrg") {
     println("svrg is running")


     val sgd = new LogisticRegressionWithSGDSVRG2(1.0, iteration, 1, sampleFraction.toDouble,sampleForSVRG.toDouble)
     val model = sgd.run(training)
     model.clearThreshold()
     val predictionAndLabels = test.map { case LabeledPoint(label, features) =>
       val prediction = model.predict(features)
       (label, prediction)
     }

     val metric = new BinaryClassificationMetrics(predictionAndLabels)
     val auROC = metric.areaUnderROC()
     println("AREA under ROC = " + auROC)

    // val model = sgd.run(training)





     println(data.count())
     val end = System.nanoTime()

     println("this is time" + (end - start) / 1000000)
   }
//    val sgd = new LogisticRegressionWithSGDSVRG(1.0, 1, 1, 1)

//    val sgd = new LogisticRegressionWithSGD(1.0, 50, 0.01, 1.0)
    //   val sgd = new LogisticRegressionWithSGDMomentum(1.0, 50, 0.01, 1.0)

         val sgdwithadagram = new LogisticRegressionWithAdagram(1.0, 50, 0.01, 1.0)

//    val sgd = new LogisticRegressionWithSGDSVRG(1.0, 50, 0.01, 1.0)

    // val model = sgd.run(training)
    val model = sgdwithadagram.run(training)

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