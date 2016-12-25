package org.apache.spark.mllib.optimization.adagram

/**
  * Created by kly on 2016/12/8.
  */

import breeze.linalg.{DenseVector, Vector => BV, axpy => brzAxpy, norm => brzNorm}
import org.apache.spark.mllib.linalg.{Vector, Vectors}


/**
  * Created by kly on 2016/12/8.
  */
class adagram {
  def compute(
               weightsOld: Vector,
               gradient: Vector,
               stepSize: Double,
               iter: Int,
               regParam: Double,
               eta: DenseVector[Double]): (Vector, Double) = {
    // add up both updates from the gradient of the loss (= step) as well as
    // the gradient of the regularizer (= regParam * weightsOld)
    // w' = w - thisIterStepSize * (gradient + regParam * w)
    // w' = (1 - thisIterStepSize * regParam) * w - thisIterStepSize * gradient
    val thisIterStepSize = stepSize / math.sqrt(iter)
    val brzWeights: BV[Double] = weightsOld.asBreeze.toDenseVector
    //brzWeights :*= (1.0 - thisIterStepSize * regParam)
    val n = weightsOld.size
    for (i <- 0 to n-1) {
      brzWeights(i) = brzWeights(i) * (1 - thisIterStepSize * regParam * eta(i))
      brzWeights(i) = brzWeights(i) - thisIterStepSize * gradient(i) * eta(i)
    }
    //brzAxpy(-thisIterStepSize, gradient.asBreeze, brzWeights)
    val norm = brzNorm(brzWeights, 2.0)
    println("this is my adagram")
    (Vectors.fromBreeze(brzWeights), 0.5 * regParam * norm * norm)

  }
}