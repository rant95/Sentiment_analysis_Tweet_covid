// Databricks notebook source
// MAGIC %md
// MAGIC # Project Tweets Covid19 prediction CV
// MAGIC 
// MAGIC ### Naives Bayes Cross Validation

// COMMAND ----------

import com.mongodb.spark._
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.functions._
import com.databricks.spark.corenlp.functions._
import org.apache.spark._

import scala.collection.mutable
import org.apache.spark.mllib.clustering.LDA
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD

import org.apache.spark.sql.{Row, SQLContext}
import org.apache.spark.sql.types.{StructField, StructType}
import org.apache.spark.sql.types.DoubleType

import org.apache.spark.sql.functions.{from_unixtime, unix_timestamp, window}
import org.apache.spark.sql.types.{StructType, StructField, StringType}
import org.apache.spark.sql.functions._
import sqlContext.implicits._
import org.apache.spark.ml.feature.StopWordsRemover

import java.sql.Timestamp
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.types.IntegerType
import org.apache.spark.sql.functions.to_timestamp

import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator  
import org.apache.spark.mllib.evaluation.MulticlassMetrics  
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics  
import org.apache.spark.ml.classification.RandomForestClassifier  
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit, CrossValidator}  
import org.apache.spark.ml.feature.{VectorAssembler, StringIndexer, OneHotEncoderEstimator}  

// COMMAND ----------

//val tweetFe = spark.table("data_trainv0")//.withColumn("Feels",col("Feelings").cast(DoubleType))  
//val tweetTe = spark.table("data_testv0")//.withColumn("Feels",col("Feelings").cast(DoubleType)) 

val tweetFe = spark.read.format("com.mongodb.spark.sql.DefaultSource").option("database","allnews").option("collection","data_trainv0").load()
val tweetTe = spark.read.format("com.mongodb.spark.sql.DefaultSource").option("database","allnews").option("collection","data_testv0").load()
val tweetFeel=tweetFe.withColumnRenamed("ctext","OriginalTweet")
val tweetTest =tweetTe.withColumnRenamed("text","OriginalTweet")

tweetFeel.printSchema()
tweetTest.printSchema()

// COMMAND ----------

tweetFeel.show(5) 

// COMMAND ----------

tweetTest.show(5) 

// COMMAND ----------

// MAGIC %md
// MAGIC ### 1. Tweet to Lemmetization

// COMMAND ----------


val withLemmes = tweetFeel.withColumn("lemmas", lemma(col("OriginalTweet")))
val withLemmesString = withLemmes.withColumn("lemmas", concat_ws(" ",col("lemmas"))) //pour passer de lemme  tokenization
//withLemmesString.show(7)
//val withLemmesDF = withLemmes.select("Feelings","lemmas")
//val withLemmesRDD = withLemmesDF.rdd.map(f => (f.getAs[String](0),f.getAs[String](1)))

//Test data

val withLemmesTest = tweetTest.withColumn("lemmas", lemma(col("OriginalTweet")))
val withLemmesStringTest = withLemmesTest.withColumn("lemmas", concat_ws(" ",col("lemmas"))) //pour passer de lemme  tokenization

// COMMAND ----------

/*display(withLemmes
    .select(explode(col("lemmas")).as("lemmas"))
    .groupBy("lemmas")
    .count()
    .orderBy(col("count").desc).limit(10))*/
    

// COMMAND ----------

// MAGIC %md
// MAGIC ### 1.2 Delete stop word

// COMMAND ----------

import org.apache.spark.ml.feature.StopWordsRemover

// COMMAND ----------

val remover = new StopWordsRemover()
    .setStopWords(StopWordsRemover.loadDefaultStopWords("english") 
        ++ Array("!", "?", "@", "...", "'", ",", ".", "!","!!", "_","??","???", ":", ";", "-LRB-", "-RRB-", "#", "`","'s","-","&","''","``","n't","‰","¥","˜","ð","®","¹","¦","¶","•","-lsb-","-rsb-","amp"))
    .setInputCol("lemmas")
    .setOutputCol("filtered_lemmas")

val LemmeswithoutStopwords = remover.transform(withLemmes)

    

// COMMAND ----------

val LemmeswithoutStopwordsTest = remover.transform(withLemmesTest)

// COMMAND ----------

// MAGIC %md
// MAGIC ### 1.2.1 Counted Lemme

// COMMAND ----------

/*display(LemmeswithoutStopwords
    .select(explode(col("filtered_lemmas")).as("filtered_lemmas"))
    .groupBy("filtered_lemmas")
    .count()
    .orderBy(col("count").desc).limit(10))*/

// COMMAND ----------

// MAGIC %md
// MAGIC ### 1.3 Lemmetization to Tokenization

// COMMAND ----------


val LemmeswithoutStopwordsString = LemmeswithoutStopwords.withColumn("filtered_lemmas", concat_ws(" ",col("filtered_lemmas"))) //pour passer de lemme  tokenization


//val withTokens = LemmeswithoutStopwordsString.withColumn("tokens", tokenize(col("filtered_lemmas")))

//withTokens.show(7)

//test data

val LemmeswithoutStopwordsStringTest = LemmeswithoutStopwordsTest.withColumn("filtered_lemmas", concat_ws(" ",col("filtered_lemmas"))) //pour passer de lemme  tokenization


//val withTokensTest = LemmeswithoutStopwordsStringTest.withColumn("tokens", tokenize(col("filtered_lemmas")))

// COMMAND ----------

// MAGIC %md
// MAGIC ### 1.3.1 Counted Tokens

// COMMAND ----------

/*display(withTokens
    .select(explode(col("tokens")).as("token"))
    .groupBy("token")
    .count()
    .orderBy(col("count").desc)
    .limit(15))*/

// COMMAND ----------

// MAGIC %md
// MAGIC ### 2.1 Tweet to tokenization

// COMMAND ----------

//val withTokensSW = tweetFeel.withColumn("tokens", tokenize(col("OriginalTweet")))

// COMMAND ----------



/*display(withTokensSW .select(explode(col("tokens")).as("token")).groupBy("token").count().orderBy(col("count").desc).limit(15))*/

// COMMAND ----------

// MAGIC %md
// MAGIC ### 2.2 Remove Stop words

// COMMAND ----------

/*import org.apache.spark.ml.feature.StopWordsRemover

val remover = new StopWordsRemover()
    .setStopWords(StopWordsRemover.loadDefaultStopWords("english") 
        ++ Array("!", "?", "@", "...", "'", ",", ".", "!", "_", "?", ":", ";", "-LRB-", "-RRB-", "#", "`","'s","-","&","''","``","n't","get"))
    .setInputCol("tokens")
    .setOutputCol("filtered_tokens")

val withoutStopwords = remover.transform(withTokensSW)*/


// COMMAND ----------

// MAGIC %md
// MAGIC ### 2.3 Counted Tokens

// COMMAND ----------

/*display(withoutStopwords
    .select(explode(col("filtered_tokens")).as("token"))
    .groupBy("token")
    .count()
    .orderBy(col("count").desc)
    .limit(15))*/




// COMMAND ----------

// MAGIC %md
// MAGIC ###TF-IDF Calcul de la matrice TF-IDF

// COMMAND ----------

import org.apache.spark.ml.feature.{RegexTokenizer, Tokenizer}
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer}
val tokenizer = new Tokenizer().setInputCol("filtered_lemmas").setOutputCol("tokens")
val regexTokenizer = new RegexTokenizer()
  .setInputCol("filtered_lemmas")
  .setOutputCol("tokens")
  .setPattern("\\W")
// alternatively .setPattern("\\w+").setGaps(false)

//val countTokens = udf { (words: Seq[String]) => tokens.length }

val withTokens = tokenizer.transform(LemmeswithoutStopwordsString)
val withTokensTest = tokenizer.transform(LemmeswithoutStopwordsStringTest)

// COMMAND ----------

val hashingTF = new HashingTF()
  .setInputCol("tokens").setOutputCol("rawFeatures").setNumFeatures(2000)

// COMMAND ----------

//wordsData.show(10)

// COMMAND ----------

val featurizedData = hashingTF.transform(withTokens)
//featurizedData.show(10)

val featurizedDataTest = hashingTF.transform(withTokensTest)

// COMMAND ----------

//val featurizedDataRDD = featurizedData.select("rawFeatures").rdd.map(_.getAs[org.apache.spark.ml.linalg.Vector](0)).map(org.apache.spark.mllib.linalg.Vectors.fromML)

// COMMAND ----------

// MAGIC %md
// MAGIC ### Count Vector

// COMMAND ----------

// alternatively, CountVectorizer can also be used to get term frequency vectors

/*val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")
val idfModel = idf.fit(featurizedData)

val rescaledData = idfModel.transform(featurizedData)
rescaledData.select("Location","feelings", "features").show()*/

// COMMAND ----------

import org.apache.spark._
import org.apache.spark.rdd._

//val tweet= ratingsDS.select("OriginalTweet")

//val tweetRDD: org.apache.spark.rdd.RDD[org.apache.spark.sql.Row] = tweet.rdd
/*val rescaledData2 = rescaledData.select("Feelings", "features")
val rescaledDataRDD = rescaledData2.rdd.map(f => (f.getAs[String](0),f.getAs[String](1)))*/

// COMMAND ----------


featurizedData.select("Feelings","rawFeatures").printSchema()
featurizedDataTest.select("Feelings","rawFeatures").printSchema()

// COMMAND ----------

val featurizedData2=featurizedData.select("Feelings","rawFeatures","OriginalTweet","Sentiments")
val featurizedData2Test=featurizedDataTest.select("Feelings","rawFeatures","OriginalTweet","Sentiments")


// COMMAND ----------

//featurizedData2.show()

// COMMAND ----------

// MAGIC %md
// MAGIC ### Naives Bayes CV

// COMMAND ----------

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

// COMMAND ----------

 // columns that need to added to feature column
/*val cols = Array("rawFeatures")
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.ml.tuning.{CrossValidator, CrossValidatorModel, ParamGridBuilder}
  // VectorAssembler to add feature column
  // input columns - cols
  // feature column - features
  val assembler = new VectorAssembler()
    .setInputCols("cols")
    .setOutputCol("features")
  val featureDf = assembler.transform(featurizedData2)
  featureDf.printSchema()
  featureDf.show(10)*/

 

// COMMAND ----------

// Index labels, adding metadata to the label column.
// Fit on whole dataset to include all labels in index.
// Fit on whole dataset to include all labels in index.
val labelIndexer = new StringIndexer()
  .setInputCol("Feelings")
  .setOutputCol("indexedLabel")
 
// Automatically identify categorical features, and index them.
// Set maxCategories so features with > 4 distinct values are treated as continuous.

//val labelDf = labelIndexer.fit(featurizedData2)//.transform(featurizedData2)
val labelDf = labelIndexer.fit(featurizedData2).transform(featurizedData2)

// COMMAND ----------


//val labelDf = labelIndexer.fit(featurizedData2)//.transform(featurizedData2)
val labeltestDf = labelIndexer.fit(featurizedData2Test).transform(featurizedData2Test)

// COMMAND ----------

labeltestDf.printSchema()
labelDf.show(10)

// COMMAND ----------

  // training data set - 70%
  // test data set - 30%
  val seed = 5043
  val Array(trainingData0, testData0) = labelDf.randomSplit(Array(0.9, 0.1), seed)
  val Array(trainingData1, testData1) = labeltestDf.randomSplit(Array(0.8, 0.2), seed)



// COMMAND ----------

val Data=trainingData0.union(trainingData1)
val test=testData0.union(testData1)
 
   val Array(trainingData, testData) = Data.randomSplit(Array(0.8, 0.2), seed)
val labeltestDf =test

// COMMAND ----------

trainingData.cache()
testData.cache()
labeltestDf.cache()

// COMMAND ----------

import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit, CrossValidator}  

// create the model
val naiveBayes = new NaiveBayes().setLabelCol("indexedLabel")
  .setFeaturesCol("rawFeatures")
   

// create the param grid
val paramGrid = new ParamGridBuilder()
   .addGrid(naiveBayes.smoothing, Array(0.8, 0.3, 1))
   //.addGrid(naiveBayes.modelType,Array("bernoulli","multinomial"))
   .build()

// create cross val object, define scoring metric
val cv = new CrossValidator().
  setEstimator(naiveBayes).
  setEvaluator(new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")).
  setEstimatorParamMaps(paramGrid).
  setNumFolds(2)
  
  

// You can then treat this object as the model and use fit on it.
val CVmodel = cv.fit(trainingData)



// COMMAND ----------

val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")

// COMMAND ----------


// test model with validation data
val cvPredictionDf = CVmodel.transform(testData)
//cvPredictionTestDf.show(10)
val cvAccuracy = evaluator.evaluate(cvPredictionDf)
println(cvAccuracy)

// COMMAND ----------

// test model with test data
val cvPredictionTestDf = CVmodel.transform(labeltestDf)
//cvPredictionTestDf.show(10)
val cvAccuracyTest = evaluator.evaluate(cvPredictionTestDf)
println(cvAccuracyTest)

// COMMAND ----------

CVmodel.avgMetrics


// COMMAND ----------

CVmodel.getEstimatorParamMaps

// COMMAND ----------

CVmodel.params.toList.mkString("(", ",", ")")


// COMMAND ----------

CVmodel.explainParams()


// COMMAND ----------

CVmodel.bestModel.extractParamMap().toString()

// COMMAND ----------

val predictionAndLabelsTest=cvPredictionTestDf.select("prediction","indexedLabel").rdd.map(f => (f.getAs[Double](0),f.getAs[Double](1)))

// Instantiate a new metrics objects
//val bMetrics = new BinaryClassificationMetrics(predictionAndLabels)
val mMetrics = new MulticlassMetrics(predictionAndLabelsTest)
val labels = mMetrics.labels

// Print out the Confusion matrix
println("Confusion matrix:")
println(mMetrics.confusionMatrix)


// Precision by label
labels.foreach { l =>
  println(s"Precision($l) = " + mMetrics.precision(l))
}

// Recall by label
labels.foreach { l =>
  println(s"Recall($l) = " + mMetrics.recall(l))
}

// False positive rate by label
labels.foreach { l =>
  println(s"FPR($l) = " + mMetrics.falsePositiveRate(l))
}

// F-measure by label
labels.foreach { l =>
  println(s"F1-Score($l) = " + mMetrics.fMeasure(l))
}



// COMMAND ----------

val predictionAndLabels=cvPredictionDf.select("prediction","indexedLabel").rdd.map(f => (f.getAs[Double](0),f.getAs[Double](1)))

// Instantiate a new metrics objects
//val bMetrics = new BinaryClassificationMetrics(predictionAndLabels)
val mMetrics = new MulticlassMetrics(predictionAndLabels)
val labels = mMetrics.labels

// Print out the Confusion matrix
println("Confusion matrix:")
println(mMetrics.confusionMatrix)


// Precision by label
labels.foreach { l =>
  println(s"Precision($l) = " + mMetrics.precision(l))
}

// Recall by label
labels.foreach { l =>
  println(s"Recall($l) = " + mMetrics.recall(l))
}

// False positive rate by label
labels.foreach { l =>
  println(s"FPR($l) = " + mMetrics.falsePositiveRate(l))
}

// F-measure by label
labels.foreach { l =>
  println(s"F1-Score($l) = " + mMetrics.fMeasure(l))
}


// COMMAND ----------

val bMetrics = new BinaryClassificationMetrics(predictionAndLabelsTest)
// Precision by threshold
val precision = bMetrics.precisionByThreshold
precision.foreach { case (t, p) =>
  println(s"Threshold: $t, Precision: $p")
}

// Recall by threshold
val recall = bMetrics.recallByThreshold
recall.foreach { case (t, r) =>
  println(s"Threshold: $t, Recall: $r")
}

// Precision-Recall Curve
val PRC = bMetrics.pr

// F-measure
val f1Score = bMetrics.fMeasureByThreshold
f1Score.foreach { case (t, f) =>
  println(s"Threshold: $t, F-score: $f, Beta = 1")
}

val beta = 0.5
val fScore = bMetrics.fMeasureByThreshold(beta)
f1Score.foreach { case (t, f) =>
  println(s"Threshold: $t, F-score: $f, Beta = 0.5")
}

// AUPRC
val auPRC = bMetrics.areaUnderPR
println("Area under precision-recall curve = " + auPRC)


// Compute thresholds used in ROC and PR curves
val thresholds = precision.map(_._1)

// ROC Curve
val roc = bMetrics.roc

// AUROC
val auROC = bMetrics.areaUnderROC
println("Area under ROC = " + auROC)

// COMMAND ----------

val bMetrics = new BinaryClassificationMetrics(predictionAndLabels)
// Precision by threshold
val precision = bMetrics.precisionByThreshold
precision.foreach { case (t, p) =>
  println(s"Threshold: $t, Precision: $p")
}

// Recall by threshold
val recall = bMetrics.recallByThreshold
recall.foreach { case (t, r) =>
  println(s"Threshold: $t, Recall: $r")
}

// Precision-Recall Curve
val PRC = bMetrics.pr

// F-measure
val f1Score = bMetrics.fMeasureByThreshold
f1Score.foreach { case (t, f) =>
  println(s"Threshold: $t, F-score: $f, Beta = 1")
}

val beta = 0.5
val fScore = bMetrics.fMeasureByThreshold(beta)
f1Score.foreach { case (t, f) =>
  println(s"Threshold: $t, F-score: $f, Beta = 0.5")
}

// AUPRC
val auPRC = bMetrics.areaUnderPR
println("Area under precision-recall curve = " + auPRC)


// Compute thresholds used in ROC and PR curves
val thresholds = precision.map(_._1)

// ROC Curve
val roc = bMetrics.roc

// AUROC
val auROC = bMetrics.areaUnderROC


