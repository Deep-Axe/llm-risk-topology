// Expected descendant exposure E_q on declared lineage. Launched by scripts/run_exposure_wsl.sh.
// Each edge passes a defect with probability q. Draws hash (seed, replicate, parent, child) the same way as
// python/exposure.py, so both implementations sample identical edges and must agree exactly.
import org.apache.spark.graphx._
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._

try {
  val root = sys.env("METADATA_HDFS")
  val output = sys.env("EXPOSURE_HDFS")
  val replicates = sys.env.getOrElse("EXPOSURE_REPLICATES", "200").toInt
  val batch = sys.env.getOrElse("EXPOSURE_BATCH", "50").toInt
  val seed = sys.env.getOrElse("EXPOSURE_SEED", "42").toLong
  val relations = sys.env.getOrElse("EXPOSURE_RELATIONS", "fine_tune,adapter,merge,quantization,other").split(",").toSeq
  val qGrid = Seq(0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0)
  require(replicates >= 2 && batch >= 1, "Need at least 2 replicates and a positive batch size")

  val edges = spark.read.parquet(root + "/processed/edges")
    .filter(col("relation").isin(relations: _*))
    .select(col("parent_model_id").as("parent"), col("child_model_id").as("child")).distinct().cache()
  val nodes = edges.select(col("parent").as("model_id")).union(edges.select(col("child").as("model_id"))).distinct()
  val qs = qGrid.toDF("q")

  // Uniform [0, 1) from the first 52 bits of sha256("seed|replicate|parent|child").
  def draws(first: Long, last: Long): DataFrame = edges.crossJoin(spark.range(first, last).toDF("r"))
    .withColumn("u", conv(substring(sha2(concat_ws("|", lit(seed.toString), col("r").cast("string"),
      col("parent"), col("child")), 256), 1, 13), 16, 10).cast("long").cast("double") / math.pow(2, 52))

  // Bottleneck closure: for each (replicate, ancestor, descendant) the smallest largest draw over all paths.
  // A descendant is reached at q exactly when that value is below q.
  def closure(sampled: DataFrame): DataFrame = {
    val step = sampled.select(col("r"), col("parent").as("via"), col("child").as("desc"), col("u")).cache()
    var best = sampled.select(col("r"), col("parent").as("anc"), col("child").as("desc"), col("u").as("b"))
      .localCheckpoint()
    var frontier = best
    var rounds = 0
    while (!frontier.isEmpty) {
      rounds += 1
      require(rounds <= 1000, "Closure did not converge; check for cycles")
      val extended = frontier.withColumnRenamed("desc", "via").join(step, Seq("r", "via"))
        .select(col("r"), col("anc"), col("desc"), greatest(col("b"), col("u")).as("b"))
        .filter(col("anc") =!= col("desc"))
      val merged = best.union(extended).groupBy("r", "anc", "desc").agg(min("b").as("b")).localCheckpoint()
      frontier = merged.as("m").join(best.as("o"), Seq("r", "anc", "desc"), "left")
        .filter(col("o.b").isNull || col("m.b") < col("o.b"))
        .select(col("r"), col("anc"), col("desc"), col("m.b").as("b")).localCheckpoint()
      best = merged
    }
    step.unpersist()
    best
  }

  val sums = (0L until replicates.toLong by batch.toLong).map { first =>
    val last = math.min(first + batch, replicates.toLong)
    val counts = closure(draws(first, last)).crossJoin(qs)
      .groupBy("r", "anc", "q").agg(sum(when(col("b") < col("q"), 1L).otherwise(0L)).as("n"))
    val part = counts.groupBy("anc", "q").agg(sum("n").as("s"), sum(col("n") * col("n")).as("s2")).localCheckpoint()
    println(s"EXPOSURE_BATCH_DONE $first-$last")
    part
  }.reduce(_ union _).groupBy("anc", "q").agg(sum("s").as("s"), sum("s2").as("s2"))

  // Independent reach at q=1 with GraphX Pregel: each vertex collects the set of its descendants.
  val names = nodes.as[String].rdd.zipWithIndex().map { case (n, i) => (i, n) }.cache()
  val ids = names.map(_.swap).toDF("model_id", "vid")
  val links = edges.join(ids.withColumnRenamed("model_id", "parent").withColumnRenamed("vid", "src"), "parent")
    .join(ids.withColumnRenamed("model_id", "child").withColumnRenamed("vid", "dst"), "child")
    .select("src", "dst").as[(Long, Long)].rdd.map { case (s, d) => Edge(s, d, 0) }
  val graph = Graph(names.mapValues(_ => Set.empty[VertexId]), links)
  val reachSets = graph.pregel(Set.empty[VertexId], activeDirection = EdgeDirection.In)(
    (_, own, incoming) => own ++ incoming,
    t => if ((t.dstAttr + t.dstId).subsetOf(t.srcAttr)) Iterator.empty else Iterator((t.srcId, t.dstAttr + t.dstId)),
    _ ++ _)
  val reach = reachSets.vertices.map { case (v, set) => (v, (set - v).size.toLong) }.toDF("vid", "reach")
    .join(ids, "vid").select("model_id", "reach")

  val result = nodes.crossJoin(qs).join(sums.withColumnRenamed("anc", "model_id"), Seq("model_id", "q"), "left")
    .na.fill(0L, Seq("s", "s2")).join(reach, "model_id")
    .withColumn("e_q_mean", col("s") / replicates)
    .withColumn("e_q_se", sqrt(greatest(col("s2") / replicates - col("e_q_mean") * col("e_q_mean"), lit(0.0)) /
      (replicates - 1)))
    .select(col("model_id"), col("q"), col("reach"), col("e_q_mean"), col("e_q_se"),
      lit(replicates).as("replicates"), lit(seed).as("seed"), lit(relations.mkString(",")).as("relations"))
    .cache()

  val atOne = result.filter(col("q") === 1.0)
  val reachMismatch = atOne.filter(col("e_q_mean") =!= col("reach") || col("e_q_se") =!= 0.0).count()
  assert(reachMismatch == 0, s"Monte Carlo q=1 disagrees with Pregel reach for $reachMismatch nodes")
  assert(result.filter(col("q") === 0.0 && col("e_q_mean") =!= 0.0).count() == 0, "Nonzero exposure at q=0")
  assert(result.count() == nodes.count() * qGrid.size, "Missing node or q rows")

  result.write.mode("errorifexists").parquet(output + "/node_exposure")
  result.orderBy("model_id", "q").coalesce(1).write.option("header", "true").csv(output + "/node_exposure_csv")
  val summary = Seq((edges.count(), nodes.count(), replicates, seed, relations.mkString(","), sc.master, sc.applicationId))
    .toDF("edges", "nodes", "replicates", "seed", "relations", "spark_master", "spark_application_id")
  summary.coalesce(1).write.json(output + "/summary")
  summary.show(false)
  println("EXPOSURE_SPARK_OK")
  spark.stop()
  System.exit(0)
} catch {
  case error: Throwable => error.printStackTrace(); System.exit(1)
}
