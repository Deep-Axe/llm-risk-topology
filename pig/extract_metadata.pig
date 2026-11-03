-- Python for parsing messy JSON; Pig distributes extraction/filter/dedup on YARN.
SET default_parallel 4;
DEFINE parse_cards `python3 extract_lineage.py` SHIP ('$PARSER');
cards = LOAD '$INPUT' USING PigStorage('\t') AS (model_id:chararray, revision:chararray, card_json:chararray);
parsed = STREAM cards THROUGH parse_cards AS (status:chararray, child:chararray, parent:chararray, relation:chararray, evidence_field:chararray, evidence_value:chararray, reason:chararray);
valid = FILTER parsed BY status == 'valid';
unique_edges = DISTINCT valid;
diagnostics = FILTER parsed BY status != 'valid';
STORE unique_edges INTO '$OUTPUT/edges' USING PigStorage('\t');
STORE diagnostics INTO '$OUTPUT/diagnostics' USING PigStorage('\t');
