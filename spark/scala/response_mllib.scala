// Response-embedding graph and clusters with Spark MLlib. Launched by scripts/run_layer2_mllib_wsl.sh.
// Per model: BRP-LSH candidate neighbors (recall checked against exact search on a sample) and K-Means clusters
// with the refusal rate of each cluster. Vectors are unit length, so Euclidean distance sqrt(2 - 2 cos).
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.evaluation.ClusteringEvaluator
import org.apache.spark.ml.feature.BucketedRandomProjectionLSH
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions._

try {
  val input = sys.env("LAYER2_HDFS") + "/responses"
  val output = sys.env("LAYER2_OUTPUT")
  val k = sys.env.getOrElse("LAYER2_KNN", "10").toInt
  val clusters = sys.env.getOrElse("LAYER2_CLUSTERS", "8").toInt
  val toVector = udf((values: Seq[Double]) => Vectors.dense(values.toArray))
  val responses = spark.read.parquet(input).withColumn("features", toVector(col("vector"))).drop("vector")
    .withColumn("rid", monotonically_increasing_id()).cache()
  val models = responses.select("model_id").distinct().as[String].collect().sorted
  val summaries = scala.collection.mutable.ArrayBuffer[(String, Long, Long, Double, Double, Int, Double)]()

  for (model <- models) {
    val data = responses.filter(col("model_id") === model).cache()
    val n = data.count()
    val lsh = new BucketedRandomProjectionLSH().setBucketLength(0.6).setNumHashTables(12).setSeed(42)
      .setInputCol("features").setOutputCol("hashes").fit(data)
    val hashed = lsh.transform(data).cache()
    // Distance 1.0 is cosine 0.5; candidates beyond it are not useful as neighbors.
    val candidates = lsh.approxSimilarityJoin(hashed, hashed, 1.0, "distance")
      .select(col("datasetA.rid").as("src"), col("datasetB.rid").as("dst"), col("distance"))
      .filter(col("src") =!= col("dst"))
    val rank = Window.partitionBy("src").orderBy(col("distance"), col("dst"))
    val knn = candidates.withColumn("rank", row_number().over(rank)).filter(col("rank") <= k).cache()
    val edges = knn.count()
    // Recall against exact top-k for a fixed sample of query nodes.
    val dist = udf((a: Vector, b: Vector) => math.sqrt(Vectors.sqdist(a, b)))
    val queries = data.orderBy(hash(col("response_id"))).limit(200).select(col("rid").as("src"), col("features").as("qf"))
    val exact = queries.crossJoin(data.select(col("rid").as("dst"), col("features")))
      .filter(col("src") =!= col("dst")).withColumn("distance", dist(col("qf"), col("features")))
      .withColumn("rank", row_number().over(rank)).filter(col("rank") <= k).select("src", "dst")
    val found = exact.join(knn.select("src", "dst"), Seq("src", "dst")).count()
    val recall = found.toDouble / math.max(exact.count(), 1L)
    val kUsed = math.max(2, math.min(clusters, (n / 20).toInt))
    val kmeans = new KMeans().setK(kUsed).setSeed(42).setFeaturesCol("features").setPredictionCol("cluster")
      .fit(data)
    val assigned = kmeans.transform(data).cache()
    val silhouette = new ClusteringEvaluator().setFeaturesCol("features").setPredictionCol("cluster")
      .setDistanceMeasure("cosine").evaluate(assigned)
    assigned.select("response_id", "model_id", "prompt_id", "split", "source", "intent", "variant_type", "refusal",
      "response_harmful", "cluster").write.mode("append").parquet(output + "/clusters")
    knn.join(data.select(col("rid").as("src"), col("response_id").as("src_response_id")), "src")
      .join(data.select(col("rid").as("dst"), col("response_id").as("dst_response_id")), "dst")
      .select(lit(model).as("model_id"), col("src_response_id"), col("dst_response_id"), col("distance"), col("rank"))
      .write.mode("append").parquet(output + "/knn_edges")
    summaries += ((model, n, edges, recall, silhouette, kUsed, edges.toDouble / n))
    println(f"LAYER2_MODEL_DONE $model n=$n edges=$edges recall@$k=$recall%.3f silhouette=$silhouette%.3f")
    Seq(hashed, knn, assigned, data).foreach(_.unpersist())
  }
  val summary = summaries.toSeq.toDF("model_id", "responses", "knn_edges", "lsh_recall_at_k", "silhouette_cosine",
    "clusters", "mean_neighbors")
  summary.coalesce(1).write.mode("overwrite").json(output + "/summary")
  summary.show(false)
  println("LAYER2_MLLIB_OK")
  spark.stop()
  System.exit(0)
} catch {
  case error: Throwable => error.printStackTrace(); System.exit(1)
}
