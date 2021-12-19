// Databricks notebook source
// MAGIC %md
// MAGIC # Project Tweets Covid19 exploration
// MAGIC 
// MAGIC ### Lemme to tokenization

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

val tweetFeel = spark.table("data_train")//.withColumn("Feels",col("Feelings").cast(DoubleType))  
val tweetTest = spark.table("data_test")//.withColumn("Feels",col("Feelings").cast(DoubleType)) 

//val tweetFe = spark.read.format("com.mongodb.spark.sql.DefaultSource").option("database","allnews").option("collection","data_train").load()
//val tweetTe = spark.read.format("com.mongodb.spark.sql.DefaultSource").option("database","allnews").option("collection","data_test").load()
val tweetFeel=tweetFe.withColumnRenamed("ctext","OriginalTweet")
val tweetTest =tweetTe.withColumnRenamed("text","OriginalTweet")
tweetFeel.printSchema()
tweetTest.printSchema()

// COMMAND ----------


val withLemmes = tweetFeel.withColumn("lemmas", lemma(col("OriginalTweet")))
val withLemmesString = withLemmes.withColumn("lemmas", concat_ws(" ",col("lemmas"))) //pour passer de lemme  tokenization
//display(withLemmesString.limit(10))
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
display(LemmeswithoutStopwords.limit(10))

// COMMAND ----------

LemmeswithoutStopwords
    .select(explode(col("filtered_lemmas")).as("filtered_lemmas"))
    .groupBy("filtered_lemmas")
    .count()
    .orderBy(col("count").desc).count()

// COMMAND ----------

LemmeswithoutStopwords
    .select(explode(col("filtered_lemmas")).as("filtered_lemmas"))
    .groupBy("filtered_lemmas")
    .count()
    .orderBy(col("count").desc).show(10)

// COMMAND ----------

// MAGIC %md
// MAGIC ### 2.1 Lemme to tokenization

// COMMAND ----------


val LemmeswithoutStopwordsString = LemmeswithoutStopwords.withColumn("filtered_lemmas", concat_ws(" ",col("filtered_lemmas"))) //pour passer de lemme  tokenization


val withTokens = LemmeswithoutStopwordsString.withColumn("tokens", tokenize(col("filtered_lemmas")))

display(withTokens.limit(10))


// COMMAND ----------

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

// COMMAND ----------

tokenized.select("filtered_lemmas", "words").withColumn("tokens", countTokens(col("words"))).show(10)

// COMMAND ----------

tokenized.select(explode(col("words")).as("token"))
    .groupBy("token")
    .count()
    .orderBy(col("count").desc)
    .count()

// COMMAND ----------

tokenized.select(explode(col("words")).as("token"))
    .groupBy("token")
    .count()
    .orderBy(col("count").desc)
    .show(10)

// COMMAND ----------

/*withTokens
    .select(explode(col("tokens")).as("token"))
    .groupBy("token")
    .count()
    .orderBy(col("count").desc)
    .show(10)*/

// COMMAND ----------

// MAGIC %md
// MAGIC ### N grams

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

display(withTrigram.select("ngrams").limit(10))

// COMMAND ----------

display(withBigram.select("ngrams").limit(5))

// COMMAND ----------

withBigram
    .select(explode(col("ngrams")).as("token"))
    .groupBy("token")
    .count()
    .orderBy(col("count").desc)
    .show(10)

// COMMAND ----------

withTrigram
    .select(explode(col("ngrams")).as("token"))
    .groupBy("token")
    .count()
    .orderBy(col("count").desc)
    .show(10)

// COMMAND ----------

val withPos = tokenized.withColumn("pos", pos(col("filtered_lemmas")))
//val withNe = tokenized.withColumn("ne", ner(col("OriginalTweet")))

// COMMAND ----------

withPos.show(10)

// COMMAND ----------

//withNe.show(10)

// COMMAND ----------

display(withBigram
    .select(explode(col("ngrams")).as("token"))
    .groupBy("token")
    .count()
    .orderBy(col("count").desc)
    .limit(10))

// COMMAND ----------

display(withTrigram
    .select(explode(col("ngrams")).as("token"))
    .groupBy("token")
    .count()
    .orderBy(col("count").desc)
    .limit(10))
