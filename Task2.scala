import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
import org.apache.spark.mllib.clustering.BisectingKMeans
import org.apache.spark.mllib.feature.{HashingTF, IDF}
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import java.io.{BufferedWriter, File, FileWriter}
import scala.util.parsing.json.JSONObject

object Task2 {

  def main(args: Array[String]): Unit = {

    val sconf = new SparkConf().setAppName("Mahima_Gupta_hw4").setMaster("local[*]").set("spark.driver.host", "localhost")
    val sctxt = new SparkContext(sconf)

    val input = sctxt.textFile(args(0))
    //val input = sctxt.textFile("/Users/mahima/IdeaProjects/DMAssignment4/src/main/scala/yelp_reviews_clustering_small.txt")
    val seed = 42
    val algo = args(1)
    //val algo = "B"
    val num_clusters =args(2).toInt
    //val num_clusters = 8
    val num_iterations = args(3).toInt
    //val num_iterations = 20

    val documents: RDD[Seq[String]] = input.map(_.split(" ").toSeq)

    //println(documents.count())
    val output_file ="Mahima_Gupta_Cluster_result.json"
    //val output_file = "/Users/mahima/IdeaProjects/DMAssignment4/src/main/scala/Mahima_Gupta_Cluster_B_8_20.json"

    val hashingTF = new HashingTF()
    val tf: RDD[Vector] = hashingTF.transform(documents)

    tf.cache()
    val idf = new IDF().fit(tf)
    val tfidf: RDD[Vector] = idf.transform(tf)
    tfidf.cache()

    tfidf.take(5).foreach(println("tfidf", _))

    val output_file_writer = new BufferedWriter(new FileWriter(new File(output_file)))
    output_file_writer.write("{")
    output_file_writer.flush()


    if (algo == "K")
    {
      output_file_writer.append("\"algorithm\" : \"K-Means\",")
      output_file_writer.append("\n\n")

      println("Running K-Means")
      val clusters = KMeans.train(tfidf, num_clusters, num_iterations, "random", 42)
      //println(clusters)
      val cluster_centers = clusters.clusterCenters
      val predict = clusters.predict(tfidf)
      //predict.foreach(println)
      val features = tfidf.zip(documents).zip(predict).collectAsMap()
      val out = features.map(row => (((row._1._2, row._1._1), Vectors.sqdist(row._1._1, clusters.clusterCenters(row._2))), row._2))
      out.take(10).foreach(row => println(row._1._1._2))

      val WSSSE = clusters.computeCost(tfidf)
      output_file_writer.append("\"WSSSE\" : \"" + WSSSE + "\",")

      val clustersout = Array[JSONObject]()
      var WSSSE2 = 0.0

      output_file_writer.append("\n\n")
      output_file_writer.append("\"clusters\" : [ ")

      for (i <- 0 to num_clusters - 1)
      {

        val clusterone = out.filter(row => row._2 == i).toSeq
        var error_clusterone = 0.0
        var wordsone = List[String]()
        clusterone.foreach(row => {
          error_clusterone = error_clusterone + row._1._2
          wordsone = wordsone ::: row._1._1._1.toList

        })
        WSSSE2 += error_clusterone
        val top10words = wordsone.groupBy(identity).toList.sortBy(_._2.size).reverse.take(10).map(_._1).toArray
        //println(top10)
        //top10.foreach(println)
        output_file_writer.append("\n\n")
        output_file_writer.append("{ \"id\" : " + (i+1) + ",")
        output_file_writer.append("\n")
        output_file_writer.append("\"size\" : " + clusterone.length + ",")
        output_file_writer.append("\n")
        output_file_writer.append("\"error\" : " + error_clusterone + ",")
        output_file_writer.append("\n")
        output_file_writer.append("\"terms\" : [ ")
        var top10strings = top10words.mkString("\",\"")
        output_file_writer.append("\"" + top10strings + "\"")
        output_file_writer.append(" ] }")
        //println("cluster_1 size", i, cluster_1.length, cluster_1_error)
        println(i, clusterone.length, error_clusterone, WSSSE2, WSSSE)
        //val size_cluster_1 = cluster_1.size

        if (i < num_clusters-1) {
          output_file_writer.append(",")
        }
      }

      output_file_writer.append("\n\n")
      output_file_writer.write("]")
      output_file_writer.append("\n\n")
      output_file_writer.write("}")
      output_file_writer.close()
    }

    else if (algo == "B")
    {

      output_file_writer.append("\"algorithm\" : \"Bisecting K-Means\",")
      println("Running Bisecting K-Means")
      val bisectingkm = new BisectingKMeans().setK(num_clusters).setMaxIterations(num_iterations).setSeed(42)
      val clusters = bisectingkm.run(tfidf)
      val WSSSE = clusters.computeCost(tfidf)
      val centers = clusters.clusterCenters
      val predict = clusters.predict(tfidf)

      val features = tfidf.zip(documents).zip(predict).collectAsMap()
      val out = features.map(row => (((row._1._2, row._1._1), Vectors.sqdist(row._1._1, clusters.clusterCenters(row._2))), row._2))

      //val WSSSE = clusters.computeCost(tfidf)
      output_file_writer.append("\n\n")
      output_file_writer.append("\"WSSSE\" : \"" + WSSSE + "\",")

      val clustersoutput = Array[JSONObject]()
      var WSSSE2 = 0.0
      output_file_writer.append("\n\n")
      output_file_writer.append("\"clusters\" : [ ")

      for (i <- 0 to num_clusters - 1) {
        val clusterone = out.filter(row => row._2 == i).toSeq

        val top = out.filter(row => row._2 == i).map(row => row._1._1._2)
        var error_clusterone = 0.0
        var words_1 = List[String]()
        clusterone.foreach(row => {
          error_clusterone = error_clusterone + row._1._2
          words_1 = words_1 ::: row._1._1._1.toList
        })

        WSSSE2 += error_clusterone
        val top10words = words_1.groupBy(identity).toList.sortBy(_._2.size).reverse.take(10).map(_._1).toArray
        // top10.foreach(print)
        //println()
        output_file_writer.append("\n\n")
        output_file_writer.append("{ \"id\" : " + (i+1) + ",")
        output_file_writer.append("\n")
        output_file_writer.append("\"size\" : " + clusterone.length + ",")
        output_file_writer.append("\n")
        output_file_writer.append("\"error\" : " + error_clusterone + ",")
        output_file_writer.append("\n")
        output_file_writer.append("\"terms\" : [ ")
        var top10strings = top10words.mkString("\",\"")
        output_file_writer.append("\"" + top10strings + "\"")
        output_file_writer.append(" ] }")
        println(i, clusterone.length, error_clusterone, WSSSE2, WSSSE)

        //println(WSSSE2,WSSSE)

        if (i < num_clusters - 1)
        {
          output_file_writer.append(",")
        }

      }
      output_file_writer.append("\n\n")
      output_file_writer.write("]")
      output_file_writer.append("\n\n")
      output_file_writer.write("}")
      output_file_writer.close()
    }
  }
}
