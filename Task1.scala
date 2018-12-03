import org.apache.spark.{SparkConf, SparkContext}
import scala.collection.mutable
import scala.collection.mutable.ListBuffer
import java.io.{BufferedWriter, File, FileWriter}
import scala.util.parsing.json.JSONObject

object Task1 {

  def main(args:Array[String]):Unit =
  {
    val time_start = System.currentTimeMillis()

    val sconf = new SparkConf().setAppName("Mahima_Gupta_K-Means").setMaster("local[*]")
    val scntxt = new SparkContext(sconf)

    val input_data = scntxt.textFile(args(0))
    //val input_data = scntxt.textFile("/Users/mahima/IdeaProjects/DMAssignment4/src/main/scala/yelp_reviews_clustering_small.txt")
    val feature = args(1)
    //val feature = 'T'
    val num_clusters =args(2).toInt
    //val num_clusters = 5
    val num_iterations = args(3).toInt
    //val num_iterations = 20

    val uwords_frequency = input_data.flatMap{a => a.split(" ")}.map(a=>(a,1)).reduceByKey(_+_)
    val uwords_d = uwords_frequency.map{case (a,b) => (a,0.0)}.collectAsMap()
    val wc = input_data.map(a=>a.split(" ")match{case a=> a.groupBy(identity).map{case(a,b) => (a,b.length.toDouble)}}).collect()
    val wcFeatures = wc.map{case map_1 => (map_1.keySet ++ uwords_d.keySet).map {i=> (i,map_1.getOrElse(i,0.0) + uwords_d.getOrElse(i,0.0))}.toList.sorted.map((_._2))}

    val tf = wcFeatures.map(features =>
    {
      val fsum = features.sum
      features.map(_ /fsum)
    })

    val reviews_total = wc.length
    val df  = input_data.map(x=>x.split(" ")match{ case x=>x.distinct.map((_,1.0))}).flatMap(x=>x).reduceByKey(_+_).sortByKey().map(_._2).collect()

    val tf_idf = tf.map( features =>
    {
      val zf = features zip df
      zf.map( {case (a,b)  => (a  * math.log(reviews_total / b))})
    })
    val namesOfFeature = (uwords_d.keySet ++ uwords_d.keySet).toList.sorted
    val output_file = "Mahima_Gupta_KMeans_small_result.json"
    //val output_file = "/Users/mahima/IdeaProjects/DMAssignment4/src/main/scala/Mahima_Gupta_KMeans_small_T_5_20.json"
    val output_file_writer = new BufferedWriter(new FileWriter(new File(output_file)))
    output_file_writer.write("{")
    output_file_writer.flush()

    output_file_writer.append("\"algorithm\" : \"K-Means\",")
    output_file_writer.append("\n\n")

    var result_kMeans = (new ListBuffer[List[Double]].toList,new ListBuffer[Double].toList)

    if (feature == 'W')
      result_kMeans = KMeans(wcFeatures,num_clusters,num_iterations)
    else
      result_kMeans = KMeans(tf_idf,num_clusters,num_iterations)

    val cfeatures = result_kMeans._1
    val cSSE = result_kMeans._2

    val WSSE1= cSSE.sum
    output_file_writer.append("\"WSSSE\" : \"" + WSSE1 + "\",")
    output_file_writer.append("\n\n")
    output_file_writer.append("\"clusters\" : [ ")
    output_file_writer.append("\n\n")

    var index=0
    var newSize: List[Double] = List()

    while(index < cSSE.size)
    {
      if (cSSE(index) == 0.0) newSize = newSize ::: List(1.0)
      else newSize = newSize ::: List((cSSE(index)*1000.0/WSSE1).floor)
      index += 1
    }

    var maxindex = cSSE.indexOf(cSSE.max)
    newSize = newSize.updated(maxindex, newSize(maxindex) + 1000 - newSize.sum)
    
    var i =0
    cfeatures.foreach(
      {
        vector_feature =>
          val SSE = cSSE(i)
          output_file_writer.append("{ \"id\" : " + (i+1) + ",")
          output_file_writer.append("\n")
          output_file_writer.append("\"size\" : " + newSize(i).toInt + ",")
          output_file_writer.append("\n")
          output_file_writer.append("\"error\" : " + cSSE(i) + ",")
          output_file_writer.append("\n")
          output_file_writer.append("\"terms\" : [ ")
          var top10words = namesOfFeature.zip(vector_feature).sortBy(_._2).reverse.map(_._1).take(10).mkString("\",\"")
          output_file_writer.append("\"" + top10words + "\"")
          output_file_writer.append(" ] }")
          if (i < num_clusters - 1) {
            output_file_writer.append(",")
          }
          output_file_writer.append("\n\n")
          i += 1
      }
    )
    output_file_writer.write("]")
    output_file_writer.append("\n\n")
    output_file_writer.write("}")
    output_file_writer.close()

    println("WSSE :"+ WSSE1)
    println("Total Time taken :"+(System.currentTimeMillis() - time_start)/1000)


  }

  // Eucluidean Distance Calculation
  def findED(v1:List[Double],v2:List[Double]): Double =
  {
    val zVector = v1 zip v2
    val d = zVector.map({case(a,b) => (a-b)*(a-b)}).sum
    return math.sqrt(d)
  }

  // K-Means
  def KMeans(feature_in:Array[List[Double]],cluster_num:Int,iterations_max:Int):(List[List[Double]],List[Double]) =
  {
    val indexedInput = feature_in.zipWithIndex.map({case (a,b) => (b,a)}).toMap

    val r = new scala.util.Random()
    r.setSeed(20181031)


    var centroids = r.shuffle(feature_in.toList).take(cluster_num).to[ListBuffer]

    var centroid_mem : mutable.HashMap[Int,mutable.ListBuffer[(Double,Int)]] = new mutable.HashMap[Int,mutable.ListBuffer[(Double,Int)]]()
    
    for(k <- 0 to cluster_num-1)
      centroid_mem+=(k->new ListBuffer[(Double,Int)])
    
    for (iter<- 1 to iterations_max)
     {
      var centroid_updated : mutable.HashMap[Int,mutable.ListBuffer[(Double,Int)]] = new mutable.HashMap[Int,mutable.ListBuffer[(Double,Int)]]()

      for(key <- indexedInput.keySet)
      {
        var d_smallest = Double.PositiveInfinity
        var clusterId = 0

        for(k<-0 to cluster_num-1)
        {
          val d = findED(centroids(k),indexedInput(key))
          if(d<d_smallest)
          {
            d_smallest =d
            clusterId = k
          }
        }

        if (!centroid_updated.contains(clusterId))
          centroid_updated += (clusterId-> new ListBuffer[(Double,Int)])
        centroid_updated(clusterId) += ((d_smallest,key))
      }

      if(centroid_updated==centroid_mem)
        return (centroids.toList,findSSE(centroid_mem))
      
      for (k <- 0 to cluster_num-1)
      {
        if (centroid_updated.contains(k))
        {

          centroid_mem(k) = centroid_updated(k)
        }
        centroids(k) = findCentroid(centroid_mem(k), indexedInput)

      }
    }

    return  (centroids.toList,findSSE(centroid_mem))

  }

  //Finding centroids
  def findCentroid(candidates_centroid :mutable.ListBuffer[(Double,Int)],indexedInput: Map[Int,List[Double]]):List[Double] =
  {
    if(candidates_centroid.length==0)
      return candidates_centroid.map(_._1).toList
    var vector_output = indexedInput(candidates_centroid(0)._2)
    val countOfPoints = candidates_centroid.length.toDouble
    
    for(i <- 1 to candidates_centroid.length-1)
    {
      val zCentroid = vector_output.zip(indexedInput(candidates_centroid(i)._2))
      val out = zCentroid.map({case (a,b) => (a+b)})
      vector_output = out
    }

    return vector_output.map((_.toDouble/countOfPoints)).toList
  }

  // Finding SSE
  def findSSE(candidates_centroid: mutable.HashMap[Int,mutable.ListBuffer[(Double,Int)]]): List[Double] =
  {

    val SSE = candidates_centroid.map{case(id,members) => (members.map({ case (a,b) => (a*a*1.0)}).sum)}.toList

    return SSE
  }

}
