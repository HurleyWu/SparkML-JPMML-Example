package com.livniguy

import javax.xml.transform.stream.StreamResult

import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{RandomForestClassifier, DecisionTreeClassifier}
import org.dmg.pmml.PMML
import org.jpmml.model.JAXBUtil
import org.jpmml.sparkml.ConverterUtil
import org.apache.spark.ml.feature.RFormula

/**
  * @author Guy Livni
  */
object App {

  def main(args : Array[String]) {

    Logger.getLogger("org").setLevel(Level.ERROR)
    Logger.getLogger("akka").setLevel(Level.ERROR)

    val spark = SparkSession
      .builder()
      .appName("Spark JPMML")
      //      .config("spark.sql.warehouse.dir", warehouseLocation)
      //      .enableHiveSupport()
      //      .config("spark.executor.extraClassPath", "c:\\jars\\")
      //      .config("spark.driver.extraClassPath", "c:\\jars\\")
      .master("local[2]")
      .getOrCreate()

    val df = spark.read.option("header", "true").option("inferSchema", "true").csv("Iris.csv")

    //nativeSpeaker ~ age + shoeSize + score
    val formula = new RFormula().setFormula("Species ~ .")

    //val classifier = new RandomForestClassifier()
    val classifier = new DecisionTreeClassifier()
    val pipeline = new Pipeline().setStages(Array(formula, classifier))
    val pipelineModel = pipeline.fit(df)

    //print(spark.version)

    val pmml: PMML = ConverterUtil.toPMML(df.schema, pipelineModel);

    // Viewing the result
    JAXBUtil.marshalPMML(pmml, new StreamResult(System.out));
  }
}