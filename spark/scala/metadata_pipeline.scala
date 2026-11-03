// Driver is launched by scripts/run_metadata_pipeline_wsl.sh.
// GraphX uses a simple child -> parent projection; typed edges remain preserved.
import org.apache.spark.graphx._
import org.apache.spark.sql.functions._
import java.security.MessageDigest
import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.databind.JsonNode
import scala.collection.JavaConverters._

try {
  val root = sys.env("METADATA_HDFS")
  val snapshot = sys.env("METADATA_SNAPSHOT_ID")
  val expected = sys.env("METADATA_EXPECTED_RECORDS").toLong
  val cards = spark.read.option("sep", "\t").option("quote", "\u0000")
    .schema("model_id STRING, revision STRING, card_json STRING").csv(root + "/raw/models.tsv").cache()
  assert(cards.count() == expected, "Raw card count mismatch")
  assert(cards.select("model_id").distinct().count() == expected, "Duplicate model IDs")
  cards.write.mode("errorifexists").parquet(root + "/processed/models")
  val parsed = spark.read.option("sep", "\t").option("quote", "\u0000")
    .schema("status STRING, child STRING, parent STRING, relation STRING, evidence_field STRING, evidence_value STRING, reason STRING")
    .csv(root + "/pig/edges").cache()
  assert(parsed.count() > 0, "No valid declared edges found")
  val id = udf((child: String, parent: String, relation: String) => {
    val bytes = new ObjectMapper().writeValueAsString(Array("1.0.0", snapshot, child, parent, relation)).getBytes("UTF-8")
    "e_" + MessageDigest.getInstance("SHA-256").digest(bytes).map(b => f"${b & 0xff}%02x").mkString
  })
  val edges = parsed.join(cards.select(col("model_id").as("child"), col("revision")), Seq("child"))
    .select(lit("1.0.0").as("schema_version"), id(col("child"), col("parent"), col("relation")).as("edge_id"),
      lit(snapshot).as("snapshot_id"), col("child").as("child_model_id"), col("parent").as("parent_model_id"), col("relation"),
      lit("declared").as("lineage_status"), array(struct(
        when(col("revision") === "-", concat(lit("https://huggingface.co/"), col("child")))
          .otherwise(concat(lit("https://huggingface.co/"), col("child"), lit("/blob/"), col("revision"), lit("/README.md"))).as("url"),
        when(col("revision") === "-", lit(null).cast("string")).otherwise(col("revision")).as("revision"),
        col("evidence_field").as("field"), col("evidence_value").as("value"))).as("evidence"))
    .dropDuplicates("edge_id").cache()
  assert(edges.count() == parsed.count(), "Unexpected join loss or duplicate typed edge identity")
  edges.write.mode("errorifexists").parquet(root + "/processed/edges")
  edges.coalesce(1).write.mode("errorifexists").option("ignoreNullFields", "false").json(root + "/processed/edge_contracts")
  edges.orderBy("edge_id").limit(100).coalesce(1).write.option("ignoreNullFields", "false").json(root + "/processed/edge_sample")
  val names = cards.select("model_id").union(edges.select(col("parent_model_id").as("model_id"))).distinct()
  val vertices = names.as[String].rdd.sortBy(identity).zipWithIndex().map { case (name, index) => (index + 1L, name) }.cache()
  val index = vertices.map { case (vid, name) => (name, vid) }.toDF("model_id", "vertex_id")
  val links = edges.select("child_model_id", "parent_model_id").distinct()
    .join(index.select(col("model_id").as("child_model_id"), col("vertex_id").as("src")), Seq("child_model_id"))
    .join(index.select(col("model_id").as("parent_model_id"), col("vertex_id").as("dst")), Seq("parent_model_id"))
  val graph = Graph(vertices, links.select("src", "dst").rdd.map(r => Edge(r.getLong(0), r.getLong(1), 1))).cache()
  val components = graph.connectedComponents().vertices.toDF("vertex_id", "component_id")
  val ranks = graph.staticPageRank(20).vertices.toDF("vertex_id", "upstream_pagerank")
  val incoming = graph.inDegrees.map { case (v, degree) => (v, degree.toLong) }.toDF("vertex_id", "observed_children")
  val outgoing = graph.outDegrees.map { case (v, degree) => (v, degree.toLong) }.toDF("vertex_id", "declared_parents")
  val metrics = index.join(components, Seq("vertex_id")).join(ranks, Seq("vertex_id"))
    .join(incoming, Seq("vertex_id"), "left").join(outgoing, Seq("vertex_id"), "left")
    .na.fill(0L, Seq("observed_children", "declared_parents"))
    .join(cards.select("model_id").withColumn("in_sample", lit(true)), Seq("model_id"), "left")
    .na.fill(false, Seq("in_sample")).withColumn("snapshot_id", lit(snapshot)).cache()
  assert(metrics.count() == names.count(), "Lost graph vertices")
  assert(metrics.agg(sum("observed_children")).first().getLong(0) == links.count(), "Degree sum mismatch")
  metrics.write.mode("errorifexists").parquet(root + "/processed/node_metrics")
  val diagnostics = spark.read.option("sep", "\t").option("quote", "\u0000").schema("status STRING, child STRING, parent STRING, relation STRING, evidence_field STRING, evidence_value STRING, reason STRING")
    .csv(root + "/pig/diagnostics")
  diagnostics.write.mode("errorifexists").parquet(root + "/processed/diagnostics")
  // Independent source parser: read captured API JSON, never the Pig output/TSV.
  // All corpus comparisons stay distributed; only aggregate counts leave Spark.
  val sourceText = spark.read.text(root + "/raw/models.jsonl.gz").repartition(4).cache()
  val referenceModels = sourceText.select(get_json_object(col("value"), "$.id").as("model_id"),
    coalesce(get_json_object(col("value"), "$.sha"), lit("-")).as("revision")).cache()
  assert(referenceModels.count() == expected && referenceModels.select("model_id").distinct().count() == expected)
  assert(referenceModels.exceptAll(cards.select("model_id", "revision")).count() == 0)
  assert(cards.select("model_id", "revision").exceptAll(referenceModels).count() == 0)
  val reference = sourceText.rdd.mapPartitions { rows =>
    val mapper = new ObjectMapper()
    def sorted(node: JsonNode): JsonNode = {
      if (node.isObject) {
        val result = mapper.createObjectNode()
        node.fieldNames().asScala.toSeq.sorted.foreach(k => result.set[JsonNode](k, sorted(node.get(k))))
        result
      } else if (node.isArray) {
        val result = mapper.createArrayNode()
        node.elements().asScala.foreach(n => result.add(sorted(n)))
        result
      } else node
    }
    rows.flatMap { r =>
      val doc = mapper.readTree(r.getString(0))
      val child = doc.get("id").asText()
      val revision = Option(doc.get("sha")).filter(n => !n.isNull && n.asText().nonEmpty).map(_.asText()).getOrElse("-")
      val card = Option(doc.get("cardData")).filter(_.isObject).getOrElse(mapper.createObjectNode())
      val base = card.get("base_model")
      val relation = Option(card.get("base_model_relation")).map(_.asText().toLowerCase).getOrElse("") match {
        case "finetune" | "fine-tune" | "fine_tune" => "fine_tune"
        case "quantized" | "quantization" => "quantization"
        case "adapter" => "adapter"
        case "merge" => "merge"
        case _ => "other"
      }
      if (base == null || base.isNull || (base.isArray && base.size() == 0) || (base.isTextual && base.asText() == "")) {
        Seq(("no_lineage", child, "-", "other", "base_model", "-", "missing_base_model", revision))
      } else {
        val parents = if (base.isArray) base.elements().asScala.toSeq else Seq(base)
        val evidence = mapper.writeValueAsString(sorted(base))
        parents.map { parent =>
          val name = if (parent.isTextual) parent.asText() else "-"
          val valid = parent.isTextual && name.matches("(?U)[^/\\s]+/[^/\\s]+") && name != child
          val storedName = name.replace('\t', ' ').replace('\r', ' ').replace('\n', ' ')
          (if (valid) "valid" else "rejected", child, storedName, relation, "base_model", evidence, if (valid) "-" else "invalid_or_self_parent", revision)
        }
      }
    }
  }.toDF("status", "child", "parent", "relation", "evidence_field", "evidence_value", "reason", "revision").cache()
  val expectedFlat = reference.filter(col("status") === "valid").select(col("child").as("child_model_id"), col("parent").as("parent_model_id"), col("relation"),
    col("evidence_field"), col("evidence_value"), when(col("revision") === "-", lit(null).cast("string")).otherwise(col("revision")).as("evidence_revision"),
    when(col("revision") === "-", concat(lit("https://huggingface.co/"), col("child")))
      .otherwise(concat(lit("https://huggingface.co/"), col("child"), lit("/blob/"), col("revision"), lit("/README.md"))).as("evidence_url")).distinct().cache()
  val actualFlat = edges.select(col("child_model_id"), col("parent_model_id"), col("relation"), col("evidence")(0)("field").as("evidence_field"),
    col("evidence")(0)("value").as("evidence_value"), col("evidence")(0)("revision").as("evidence_revision"), col("evidence")(0)("url").as("evidence_url"))
  val missingEdges = expectedFlat.exceptAll(actualFlat).count()
  val unexpectedEdges = actualFlat.exceptAll(expectedFlat).count()
  assert(missingEdges == 0 && unexpectedEdges == 0, "Source/Pig/Spark mismatch, including evidence")
  val requiredNulls = Seq("schema_version", "edge_id", "snapshot_id", "child_model_id", "parent_model_id", "relation", "lineage_status", "evidence").map(n => col(n).isNull).reduce(_ || _)
  val invalidContracts = edges.filter(requiredNulls || col("schema_version") =!= "1.0.0" || col("snapshot_id") =!= snapshot || col("lineage_status") =!= "declared" ||
    !col("child_model_id").rlike("^[^/\\s]+/[^/\\s]+$") || !col("parent_model_id").rlike("^[^/\\s]+/[^/\\s]+$") ||
    col("child_model_id") === col("parent_model_id") || size(col("evidence")) =!= 1 ||
    (col("evidence")(0)("revision").isNotNull && !col("evidence")(0)("revision").rlike("^[0-9a-f]{40}$")) ||
    col("edge_id") =!= concat(lit("e_"), sha2(to_json(array(lit("1.0.0"), lit(snapshot), col("child_model_id"), col("parent_model_id"), col("relation"))), 256))).count()
  assert(invalidContracts == 0, "Edge contract/identity violation")
  val referenceDiagnostics = reference.filter(col("status") =!= "valid").select("status", "child", "parent", "relation", "evidence_field", "evidence_value", "reason")
  val missingDiagnostics = referenceDiagnostics.exceptAll(diagnostics).cache()
  val unexpectedDiagnostics = diagnostics.exceptAll(referenceDiagnostics).cache()
  val diagnosticMismatchCount = missingDiagnostics.count() + unexpectedDiagnostics.count()
  if (diagnosticMismatchCount > 0) {
    missingDiagnostics.limit(20).write.json(root + "/verification_failures/missing_diagnostics")
    unexpectedDiagnostics.limit(20).write.json(root + "/verification_failures/unexpected_diagnostics")
  }
  assert(diagnosticMismatchCount == 0, "Diagnostic handoff mismatch")
  val referenceNames = referenceModels.select("model_id").union(expectedFlat.select(col("parent_model_id").as("model_id"))).distinct()
  val referenceVertices = referenceNames.as[String].rdd.sortBy(identity).zipWithIndex().map { case (name, i) => (i + 1L, name) }.cache()
  val referenceIndex = referenceVertices.map { case (vid, name) => (name, vid) }.toDF("model_id", "vertex_id")
  val referenceLinks = expectedFlat.select("child_model_id", "parent_model_id").distinct()
    .join(referenceIndex.select(col("model_id").as("child_model_id"), col("vertex_id").as("src")), Seq("child_model_id"))
    .join(referenceIndex.select(col("model_id").as("parent_model_id"), col("vertex_id").as("dst")), Seq("parent_model_id"))
  val referenceGraph = Graph(referenceVertices, referenceLinks.select("src", "dst").rdd.map(r => Edge(r.getLong(0), r.getLong(1), 1)))
  val referenceComponents = referenceGraph.connectedComponents().vertices.toDF("vertex_id", "component_id")
  assert(referenceComponents.exceptAll(metrics.select("vertex_id", "component_id")).count() == 0 &&
    metrics.select("vertex_id", "component_id").exceptAll(referenceComponents).count() == 0, "Component partition mismatch")
  assert(referenceNames.exceptAll(metrics.select("model_id")).count() == 0 && metrics.select("model_id").exceptAll(referenceNames).count() == 0)
  val relationSummary = expectedFlat.groupBy("relation").count()
  relationSummary.coalesce(1).write.json(root + "/processed/reference_relations")
  val referenceHubs = expectedFlat.select("child_model_id", "parent_model_id").distinct().groupBy("parent_model_id").count()
    .orderBy(desc("count"), asc("parent_model_id")).limit(10)
  referenceHubs.coalesce(1).write.json(root + "/processed/reference_hubs")
  val referenceCheck = Seq((true, missingEdges, unexpectedEdges, invalidContracts, expectedFlat.count(),
    reference.filter(col("status") === "no_lineage").select("child").distinct().count(), reference.filter(col("status") === "rejected").count(), reference.filter(col("status") === "valid").select("child").distinct().count(),
    referenceComponents.select("component_id").distinct().count(), referenceNames.count(),
    referenceNames.join(referenceModels.select("model_id"), Seq("model_id"), "left_anti").count()))
    .toDF("passed", "missing_edges", "unexpected_edges", "invalid_contracts", "expected_edges", "models_without_declared_lineage", "rejected_parent_entries", "models_with_valid_lineage", "weak_components", "vertices", "external_parent_vertices")
  referenceCheck.coalesce(1).write.json(root + "/processed/distributed_validation")
  val summary = Seq((snapshot, expected, edges.count(), links.count(), metrics.count(),
    metrics.filter(!col("in_sample")).count(), components.select("component_id").distinct().count(),
    parsed.select("child").distinct().count(), diagnostics.filter(col("status") === "no_lineage").select("child").distinct().count(),
    diagnostics.filter(col("status") === "rejected").count(), sc.master, sc.applicationId))
    .toDF("snapshot_id", "sampled_models", "typed_edges", "simple_edges", "vertices", "external_parent_vertices", "weak_components",
      "models_with_valid_lineage", "models_without_declared_lineage", "rejected_parent_entries", "spark_master", "spark_application_id")
  summary.coalesce(1).write.mode("errorifexists").json(root + "/processed/summary")
  summary.show(false)
  println("METADATA_PIPELINE_SPARK_OK")
  spark.stop()
  System.exit(0)
} catch {
  case error: Throwable => error.printStackTrace(); System.exit(1)
}
