package org.apache.spark.mllib.optimization.SVRG

/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import breeze.linalg.{Vector => BV, DenseVector => BDV,axpy => brzAxpy,norm => brzNorm}
import org.apache.spark.internal.Logging
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.optimization.{Gradient, Optimizer, Updater}
import org.apache.spark.rdd.RDD

import scala.math._
import scala.collection.mutable.ArrayBuffer

/**
  * Class used to solve an spark.optimization problem using Gradient Descent.
  *
  * @param gradient Gradient function to be used.
  * @param updater Updater to be used to update weights after every iteration.
  */
class GradientDescentWithSVRG2(private var gradient: Gradient, private var updater: Updater)
  extends Optimizer with Logging {

  private var stepSize: Double = 1.0
  private var numIterations: Int = 100
  private var regParam: Double = 0.0
  private var miniBatchFraction: Double = 1.0
  private var sampleForMean:Double = 1.0
  private var convergenceTol: Double = 0.0000001

  def setSampleForMean(fraction: Double): this.type = {
    require(fraction > 0 && fraction <= 1.0,
      s"Fraction for mini-batch SGD must be in range (0, 1] but got ${fraction}")
    this.sampleForMean = fraction
    this
  }

  /**
    * Set the initial step size of SGD for the first step. Default 1.0.
    * In subsequent steps, the step size will decrease with stepSize/sqrt(t)
    */
  def setStepSize(step: Double): this.type = {
    require(step > 0,
      s"Initial step size must be positive but got ${step}")
    this.stepSize = step
    this
  }

  /**
    * Set fraction of data to be used for each SGD iteration.
    * Default 1.0 (corresponding to deterministic/classical gradient descent)
    */
  def setMiniBatchFraction(fraction: Double): this.type = {
    require(fraction > 0 && fraction <= 1.0,
      s"Fraction for mini-batch SGD must be in range (0, 1] but got ${fraction}")
    this.miniBatchFraction = fraction
    println("test minibatch" + this.miniBatchFraction +":" + fraction)
    this
  }

  /**
    * Set the number of iterations for SGD. Default 100.
    */
  def setNumIterations(iters: Int): this.type = {
    require(iters >= 0,
      s"Number of iterations must be nonnegative but got ${iters}")
    this.numIterations = iters
    this
  }

  /**
    * Set the regularization parameter. Default 0.0.
    */
  def setRegParam(regParam: Double): this.type = {
    require(regParam >= 0,
      s"Regularization parameter must be nonnegative but got ${regParam}")
    this.regParam = regParam
    this
  }

  /**
    * Set the convergence tolerance. Default 0.001
    * convergenceTol is a condition which decides iteration termination.
    * The end of iteration is decided based on below logic.
    *
    *  - If the norm of the new solution vector is >1, the diff of solution vectors
    *    is compared to relative tolerance which means normalizing by the norm of
    *    the new solution vector.
    *  - If the norm of the new solution vector is <=1, the diff of solution vectors
    *    is compared to absolute tolerance which is not normalizing.
    *
    * Must be between 0.0 and 1.0 inclusively.
    */
  def setConvergenceTol(tolerance: Double): this.type = {
    require(tolerance >= 0.0 && tolerance <= 1.0,
      s"Convergence tolerance must be in range [0, 1] but got ${tolerance}")
    this.convergenceTol = tolerance
    this
  }

  /**
    * Set the gradient function (of the loss function of one single data example)
    * to be used for SGD.
    */
  def setGradient(gradient: Gradient): this.type = {
    this.gradient = gradient
    this
  }


  /**
    * Set the updater function to actually perform a gradient step in a given direction.
    * The updater is responsible to perform the update from the regularization term as well,
    * and therefore determines what kind or regularization is used, if any.
    */
  def setUpdater(updater: Updater): this.type = {
    this.updater = updater
    this
  }

  /**
    * :: DeveloperApi ::
    * Runs gradient descent on the given training data.
    * @param data training data
    * @param initialWeights initial weights
    * @return solution vector
    */

  def optimize(data: RDD[(Double, Vector)], initialWeights: Vector): Vector = {
    val (weights, _) = GradientDescentWithSVRG2.runMiniBatchSGD(
      data,
      gradient,
      updater,
      stepSize,
      numIterations,
      regParam,
      miniBatchFraction,
      sampleForMean,
      initialWeights,
      convergenceTol)
    weights
  }

}

/**
  * :: DeveloperApi ::
  * Top-level method to run gradient descent.
  */

object GradientDescentWithSVRG2 extends Logging {
  /**
    * Run stochastic gradient descent (SGD) in parallel using mini batches.
    * In each iteration, we sample a subset (fraction miniBatchFraction) of the total data
    * in order to compute a gradient estimate.
    * Sampling, and averaging the subgradients over this subset is performed using one standard
    * spark map-reduce in each iteration.
    *
    * @param data Input data for SGD. RDD of the set of data spark.examples, each of
    *             the form (label, [feature values]).
    * @param gradient Gradient object (used to compute the gradient of the loss function of
    *                 one single data example)
    * @param updater Updater function to actually perform a gradient step in a given direction.
    * @param stepSize initial step size for the first step
    * @param numIterations number of iterations that SGD should be run.
    * @param regParam regularization parameter
    * @param miniBatchFraction fraction of the input data set that should be used for
    *                          one iteration of SGD. Default value 1.0.
    * @param convergenceTol Minibatch iteration will end before numIterations if the relative
    *                       difference between the current weight and the previous weight is less
    *                       than this value. In measuring convergence, L2 norm is calculated.
    *                       Default value 0.001. Must be between 0.0 and 1.0 inclusively.
    * @return A tuple containing two elements. The first element is a column matrix containing
    *         weights for every feature, and the second element is an array containing the
    *         stochastic loss computed for every iteration.
    */
  def runMiniBatchSGD(
                       data: RDD[(Double, Vector)],
                       gradient: Gradient,
                       updater: Updater,
                       stepSize: Double,
                       numIterations: Int,
                       regParam: Double,
                       miniBatchFraction: Double,
                       sampleForMean:Double,
                       initialWeights: Vector,
                       convergenceTol: Double): (Vector, Array[Double]) = {

    val starttime = System.nanoTime()
    println("minibatch" + miniBatchFraction)

    // convergenceTol should be set with non minibatch settings
    if (miniBatchFraction < 1.0 && convergenceTol > 0.0) {
      logWarning("Testing against a convergenceTol when using miniBatchFraction " +
        "< 1.0 can be unstable because of the stochasticity in sampling.")
    }

    if (numIterations * miniBatchFraction < 1.0) {
      logWarning("Not all spark.examples will be used if numIterations * miniBatchFraction < 1.0: " +
        s"numIterations=$numIterations and miniBatchFraction=$miniBatchFraction")
    }

    val stochasticLossHistory = new ArrayBuffer[Double](numIterations)
    // Record previous weight and current one to calculate solution vector difference

    var previousWeights: Option[Vector] = None
    var currentWeights: Option[Vector] = None

    val numExamples = data.count()

    // if no data, return initial weights to avoid NaNs
    if (numExamples == 0) {
      logWarning("spark.optimization.SGD.GradientDescent.runMiniBatchSGD returning initial weights, no data found")
      return (initialWeights, stochasticLossHistory.toArray)
    }

    if (numExamples * miniBatchFraction < 1) {
      logWarning("The miniBatchFraction is too small")
    }

    // Initialize weights as a column vector
    var weights = Vectors.dense(initialWeights.toArray)
    val n = weights.size

    /**
      * For the first iteration, the regVal will be initialized as sum of weight squares
      * if it's L2 updater; for L1 updater, the same logic is followed.
      */
    var regVal = updater.compute(
      weights, Vectors.zeros(weights.size), 0, 1, regParam)._2

    var converged = false // indicates whether converged based on convergenceTol
    var i = 1
    val ss = System.nanoTime()
    println("this is svrg2 iteration" + numIterations)
    while (!converged && i <= numIterations) {

      val bcWeights = data.context.broadcast(weights)
      // Sample a subset (fraction miniBatchFraction) of the total data
      // compute and sum up the subgradients on this subset (this is one map-reduce)
      println("this is SVRG2")

      val (expectedGradientSum, elossSum, miniBatchSize) = data.sample(false,sampleForMean,42+i).treeAggregate((BDV.zeros[Double](n), 0.0, 0L))(
          seqOp = (c, v) => {
            // c: (grad, loss, count), v: (label, features)
            val l = gradient.compute(v._2, v._1, bcWeights.value, Vectors.fromBreeze(c._1))
            (c._1, c._2 + l, c._3 + 1)
          },
          combOp = (c1, c2) => {
            // c: (grad, loss, count)
            (c1._1 += c2._1, c1._2 + c2._2, c1._3 + c2._3)
          })
      if(miniBatchSize > 0){
        expectedGradientSum /= miniBatchSize.toDouble
      }
      val eGradient = data.context.broadcast(expectedGradientSum)
      val (newWeight, lossSum, numPartitions) = data.sample(false,miniBatchFraction,42+i)
        .treeAggregate((bcWeights.value.asBreeze.toDenseVector, 0.0, 1L))(
          seqOp = (c, v) => {
            // c: (grad, loss, count), v: (label, features)
            val oldGradient = BDV.zeros[Double](n)
            val l = gradient.compute(v._2, v._1, bcWeights.value, Vectors.fromBreeze(oldGradient))
            val delta = oldGradient - eGradient.value
            val newGradient = BDV.zeros[Double](n)
            val l2 = gradient.compute(v._2,v._1,Vectors.fromBreeze(c._1),Vectors.fromBreeze(newGradient))
            val finalGradient = newGradient - delta
            val brzWeights:BV[Double] = c._1
            val thisIterStepSize = stepSize / math.sqrt(i)
            brzWeights :*= (1.0 - thisIterStepSize * regParam)
            brzAxpy[Double,BV[Double],BV[Double]](-thisIterStepSize,finalGradient, brzWeights)
                          //              val norm = brzNorm(brzWeights, 2.0)
                          //              val regVal = 0.5 * regParam * norm * norm
            (brzWeights.toDenseVector, c._2 + l, 1)
          },
          combOp = (c1, c2) => {
            // c: (grad, loss, count)
            (c1._1 + c2._1, c1._2 + c2._2, c1._3 + c2._3)
          })
      if (numPartitions > 0) {
        /**
          * lossSum is computed using the weights from the previous iteration
          * and regVal is the regularization value computed in the previous iteration as well.
          */
        stochasticLossHistory += lossSum / miniBatchSize + regVal

        newWeight /= numPartitions.toDouble
        weights =  Vectors.fromBreeze(newWeight)
        val norm = brzNorm(newWeight, 2.0)
        regVal = 0.5 * regParam * norm * norm

        previousWeights = currentWeights
        currentWeights = Some(weights)
        if (previousWeights != None && currentWeights != None) {
          converged = isConverged(previousWeights.get,
            currentWeights.get, convergenceTol)
        }
      } else {
        logWarning(s"Iteration ($i/$numIterations). The size of sampled batch is zero")
      }
      val ee = System.nanoTime()
      println("compute time" + (ee - ss)/1000000)
      logWarning("compute time" + (ee - ss)/1000000)

      //      val  = data.sample(false, miniBatchFraction, 42 + i)
      //        .treeAggregate((BDV.zeros[Double](n), 0.0, 0L))(
      //          seqOp = (c, v) => {
      //            // c: (grad, loss, count), v: (label, features)
      //            val l = gradient.compute(v._2, v._1, bcWeights.value, Vectors.fromBreeze(c._1))
      //            (c._1, c._2 + l, c._3 + 1)
      //          },
      //          combOp = (c1, c2) => {
      //            // c: (grad, loss, count)
      //            (c1._1 += c2._1, c1._2 + c2._2, c1._3 + c2._3)
      //          })
      println("this is minibatch" + miniBatchSize)
      i += 1
    }

    logInfo("spark.optimization.SGD.GradientDescent.runMiniBatchSGD finished. Last 10 stochastic losses %s".format(
      stochasticLossHistory.takeRight(10).mkString(", ")))
    val endtime = System.nanoTime()
    println("time using" +  (endtime - starttime)/1000000)
    logWarning("time using" +  (endtime - starttime)/1000000)
    logInfo("iteration" +i)
    (weights, stochasticLossHistory.toArray)

  }

  /**
    * Alias of [[runMiniBatchSGD]] with convergenceTol set to default value of 0.001.
    */
  def runMiniBatchSGD(
                       data: RDD[(Double, Vector)],
                       gradient: Gradient,
                       updater: Updater,
                       stepSize: Double,
                       numIterations: Int,
                       regParam: Double,
                       miniBatchFraction: Double,
                       sampleForMean: Double,
                       initialWeights: Vector): (Vector, Array[Double]) =
  GradientDescentWithSVRG2.runMiniBatchSGD(data, gradient, updater, stepSize, numIterations,
    regParam, miniBatchFraction,sampleForMean, initialWeights, 0.00001)


  private def isConverged(
                           previousWeights: Vector,
                           currentWeights: Vector,
                           convergenceTol: Double): Boolean = {
    // To compare with convergence tolerance.
    val previousBDV = previousWeights.asBreeze.toDenseVector
    val currentBDV = currentWeights.asBreeze.toDenseVector

    // This represents the difference of updated weights in the iteration.
    val solutionVecDiff: Double = brzNorm(previousBDV - currentBDV)
    println("-----------------------------"+solutionVecDiff / Math.max(brzNorm(currentBDV),1.0))
    solutionVecDiff < convergenceTol * Math.max(brzNorm(currentBDV), 1.0)
  }

}
