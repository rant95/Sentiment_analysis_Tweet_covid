// Databricks notebook source
// MAGIC %md
// MAGIC # Project Tweets Covid19 clustering
// MAGIC 
// MAGIC ### Kmeans and Trigram

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

import java.sql.Timestamp

import org.apache.spark.sql.functions.{from_unixtime, unix_timestamp, window}
import org.apache.spark.sql.types.{StructType, StructField, StringType}
import org.apache.spark.sql.functions._
import sqlContext.implicits._
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.types.IntegerType
import org.apache.spark.sql.functions.to_timestamp

// COMMAND ----------

//val tweetFe = spark.table("data_trainv0")//.withColumn("Feels",col("Feelings").cast(DoubleType))  
//val tweetTe = spark.table("data_testv0")//.withColumn("Feels",col("Feelings").cast(DoubleType)) 

val tweetFe = spark.read.format("com.mongodb.spark.sql.DefaultSource").option("database","allnews").option("collection","data_trainv0").load()
val tweetTe = spark.read.format("com.mongodb.spark.sql.DefaultSource").option("database","allnews").option("collection","data_testv0").load()
val tweetFeel=tweetFe.withColumnRenamed("ctext","OriginalTweet")
val tweetTest =tweetTe.withColumnRenamed("ctext","OriginalTweet")
tweetFeel.printSchema()
tweetTest.printSchema()

// COMMAND ----------


val withLemmes = tweetFeel.withColumn("lemmas", lemma(col("OriginalTweet")))
val withLemmesString = withLemmes.withColumn("lemmas", concat_ws(" ",col("lemmas"))) //pour passer de lemme  tokenization
display(withLemmesString.limit(10))
//val withLemmesDF = withLemmes.select("Feelings","lemmas")
//val withLemmesRDD = withLemmesDF.rdd.map(f => (f.getAs[String](0),f.getAs[String](1)))

// COMMAND ----------

import org.apache.spark.ml.feature.StopWordsRemover

val remover = new StopWordsRemover()
    .setStopWords(StopWordsRemover.loadDefaultStopWords("english") 
        ++ Array("!", "?", "@", "...", "'", ",", ".", "!","!!", "_","??","???", ":", ";", "-LRB-", "-RRB-", "#", "`","'s","-","&","''","``","n't","‰","¥","˜","ð","®","¹","¦","¶","•","-lsb-","-rsb-","amp"))
    .setInputCol("lemmas")
    .setOutputCol("filtered_lemmas")

val LemmeswithoutStopwords = remover.transform(withLemmes)

    

// COMMAND ----------


val LemmeswithoutStopwordsString = LemmeswithoutStopwords.withColumn("filtered_lemmas", concat_ws(" ",col("filtered_lemmas"))) //pour passer de lemme  tokenization


val withTokens = LemmeswithoutStopwordsString.withColumn("tokens", tokenize(col("filtered_lemmas")))

display(withTokens.limit(10))


// COMMAND ----------

// MAGIC %md
// MAGIC ### 2.1 Tweet to tokenization

// COMMAND ----------

val withTokensSW = tweetFeel.withColumn("tokens", tokenize(col("OriginalTweet")))
import org.apache.spark.ml.feature.StopWordsRemover

val remover = new StopWordsRemover()
    .setStopWords(StopWordsRemover.loadDefaultStopWords("english") 
        ++ Array("!", "?", "@", "...", "'", ",", ".", "!","!!", "_","??","???", ":", ";", "-LRB-", "-RRB-", "#", "`","'s","-","&","''","``","n't","‰","¥","˜","ð","®","¹","¦","¶","•","-lsb-","-rsb-","amp"))
    .setInputCol("tokens")
    .setOutputCol("filtered_tokens")

val withoutStopwords = remover.transform(withTokensSW)


// COMMAND ----------

// DBTITLE 1,TF-IDF Calcul de la matrice TF-IDF
// MAGIC %md

// COMMAND ----------

import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer}
import org.apache.spark.ml.feature.{RegexTokenizer, Tokenizer}
import org.apache.spark.sql.functions._
val tokenizer = new Tokenizer().setInputCol("filtered_lemmas").setOutputCol("words")
val regexTokenizer = new RegexTokenizer()
  .setInputCol("filtered_lemmas")
  .setOutputCol("words")
  .setPattern("\\W")
// alternatively .setPattern("\\w+").setGaps(false)

val countTokens = udf { (words: Seq[String]) => words.length }

val tokenized = tokenizer.transform(LemmeswithoutStopwordsString)

val hashingTF = new HashingTF()
  .setInputCol("ngrams").setOutputCol("rawFeatures").setNumFeatures(90000)


// COMMAND ----------

//val featurizedData = hashingTF.transform(withTokens)
//display(featurizedData)

// COMMAND ----------

import org.apache.spark.ml.feature.NGram

val bigram = new NGram().setInputCol("words").setOutputCol("ngrams").setN(2)
val trigram = new NGram().setInputCol("words").setOutputCol("ngrams").setN(3)

val withBigram = bigram.transform(tokenized)
val withTrigram = trigram.transform(tokenized)

withBigram.select("ngrams").show(5, truncate = false)

// COMMAND ----------

withTrigram.select("ngrams").show(10, truncate = false)

// COMMAND ----------

val featurizedData = hashingTF.transform(withTrigram)

// COMMAND ----------

val featurizedDataRDD = featurizedData.select("rawFeatures").rdd.map(_.getAs[org.apache.spark.ml.linalg.Vector](0)).map(org.apache.spark.mllib.linalg.Vectors.fromML)

// COMMAND ----------

featurizedDataRDD.take(3).foreach(println)

// COMMAND ----------

featurizedDataRDD.cache()

// COMMAND ----------

import org.apache.spark.mllib.clustering.KMeans
        val nbClusters = 5
        val nbIterations = 50
     
        val clustering = KMeans.train(featurizedDataRDD, nbClusters, nbIterations)
        /*val outputClustering = "hdfs://head.local:9000/user/emeric/clusters"
        try { hdfs.delete(new org.apache.hadoop.fs.Path(outputClustering), true) } 
        catch { case _ : Throwable => { } }
        clustering.save(sc, outputClustering)*/
        
        val classes = clustering.predict(featurizedDataRDD)

// COMMAND ----------

val withTrigramsDF = withTrigram
    .select(explode(col("ngrams")).as("token"))
    .groupBy("token")
    .count()
    .orderBy(col("count").desc)

// COMMAND ----------

val withTrigramsDF2 = withTrigramsDF.select("count", "token")

// COMMAND ----------

withTrigramsDF2.show(10)

// COMMAND ----------

val withoutStopwordsRDD= withTrigramsDF2.rdd

// COMMAND ----------

val withoutStopwordsRDD2=withoutStopwordsRDD.collect().map(f => (f.getAs[Integer](0),f.getAs[String](1)))
//val withLemmesDF = withLemmes.select("Feelings","lemmas")
//val withLemmesRDD = withLemmesDF.rdd.map(f => (f.getAs[String](0),f.getAs[String](1)))
//.take(10).foreach(println)

// COMMAND ----------

val withoutStopwordsRDD3=withoutStopwordsRDD.collect().map(f => (f.getAs[String](1)))

// COMMAND ----------

 clustering.clusterCenters.foreach(clusterCenter => {
            val highest = clusterCenter.toArray.zipWithIndex.sortBy(-_._1).map(v => v._2).take(20)
            println("*****")
            highest.foreach { s => print( withoutStopwordsRDD3(s) + "," ) }
            println ()
            }
       )

// COMMAND ----------

//val tweetFeelRDD = tweetFeel.select("ScreenName","TweetAt","OriginalTweet").rdd.map(f => (f.getAs[Integer](0),f.getAs[Date](1),f.getAs[String](2)))
