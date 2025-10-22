import os, csv, yaml, re, google.auth
from datetime import datetime, timezone
from typing import Dict, Any, List, Tuple, Optional, Set
import json
from google.cloud import bigquery
import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig
import sys

# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================



# def _load_config(path="config.json"):
#     """
#     Load configuration from environment variable set by Airflow DAG.
#     """
#     import os
   
#     if 'BQ_OPTIMISER_CONFIG_JSON' in os.environ:
#         cfg = json.loads(os.environ['BQ_OPTIMISER_CONFIG_JSON'])
#         print("✓ Loaded configuration from Airflow Variables")
#         return cfg
#     else:
#         raise RuntimeError("Configuration not found. Must be run from Airflow DAG.")

def _load_config(path="config.json"): 
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)  
    return cfg


CFG = _load_config()
QUERY_PROJECT_ID = CFG["project"]["query_project_id"]
BQ_LOCATION = CFG["project"]["bq_location"]
VERTEX_PROJECT_ID = CFG["vertex"]["project_id"]
VERTEX_LOCATION = CFG["vertex"]["location"]
GEMINI_MODEL = CFG["vertex"]["model"]
GEMINI_ENABLED = CFG["vertex"]["enabled"]
GEMINI_TEMPERATURE = CFG["vertex"].get("temperature", 0.2)
GEMINI_TOP_P = CFG["vertex"].get("top_p", 0.9)
GEMINI_MAX_TOKENS = CFG["vertex"].get("max_output_tokens", 8192)

COST_PER_GB = CFG["cost"].get("cost_per_gb", 0.02)
COST_DECIMALS = CFG["cost"].get("cost_decimals", 6)

COMPLEXITY_MAX_LENGTH = CFG["complexity"].get("max_query_length", 3000)
COMPLEXITY_MAX_JOINS = CFG["complexity"].get("max_joins", 3)
COMPLEXITY_MAX_NESTING = CFG["complexity"].get("max_subquery_nesting", 2)
COMPLEXITY_ALLOW_EXCEPT = CFG["complexity"].get("allow_except", False)
COMPLEXITY_ALLOW_INTERSECT = CFG["complexity"].get("allow_intersect", False)
COMPLEXITY_ALLOW_UNION = CFG["complexity"].get("allow_union", False)
COMPLEXITY_ALLOW_FULL_OUTER = CFG["complexity"].get("allow_full_outer_join", False)
COMPLEXITY_ALLOW_QUALIFY = CFG["complexity"].get("allow_qualify", False)
COMPLEXITY_ALLOW_MULTI_CTE = CFG["complexity"].get("allow_multiple_ctes", False)

OPT_CHECK_SELECT_STAR = CFG["optimization"].get("check_select_star", True)
OPT_CHECK_SUBQUERY_STAR = CFG["optimization"].get("check_subquery_select_star", True)
OPT_CHECK_PARTITION = CFG["optimization"].get("check_missing_partition_filter", True)
OPT_CHECK_ORDER_BY = CFG["optimization"].get("check_order_by_no_limit", True)
SCHEMA_COLUMN_LIMIT = CFG["optimization"].get("schema_column_limit", 20)

SAVE_DEBUG_QUERIES = CFG["output"].get("save_debug_queries", True)

PROMPT_TEMPLATE = CFG["prompt"].get("system_message", "")

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def log(stage, msg, **kv):
    extras = " ".join(f"{k}={v}" for k,v in kv.items())
    print(f"[{stage}] {msg} {extras}".rstrip())

def _adc():
    creds, _ = google.auth.default()
    return creds

def _cost_from_bytes(nbytes:int)->float:
    return round((float(nbytes)/(1024**3))*COST_PER_GB, COST_DECIMALS)

# ============================================================================
# SCHEMA EXTRACTION
# ============================================================================

def extract_table_references(sql: str) -> List[Tuple[str, str, str]]:
    """
    Extract table references from SQL.
    Returns list of (project, dataset, table) tuples.
    """
    tables = []
   
    # Pattern for fully qualified tables: project.dataset.table or `project.dataset.table`
    pattern1 = r'`?([a-zA-Z0-9_-]+)\.([a-zA-Z0-9_-]+)\.([a-zA-Z0-9_-]+)`?'
   
    # Pattern for dataset.table
    pattern2 = r'FROM\s+`?([a-zA-Z0-9_-]+)\.([a-zA-Z0-9_-]+)`?'
    pattern3 = r'JOIN\s+`?([a-zA-Z0-9_-]+)\.([a-zA-Z0-9_-]+)`?'
   
    # Find fully qualified tables
    for match in re.finditer(pattern1, sql):
        project, dataset, table = match.groups()
        tables.append((project, dataset, table))
   
    # Find dataset.table patterns and use default project
    for pattern in [pattern2, pattern3]:
        for match in re.finditer(pattern, sql, re.IGNORECASE):
            dataset, table = match.groups()
            tables.append((QUERY_PROJECT_ID, dataset, table))
   
    # Remove duplicates while preserving order
    seen = set()
    unique_tables = []
    for t in tables:
        if t not in seen:
            seen.add(t)
            unique_tables.append(t)
   
    return unique_tables

def get_table_schema(project: str, dataset: str, table: str) -> Optional[Dict[str, Any]]:
    """
    Fetch schema information for a table including partition info.
    Returns dict with columns and partition info.
    """
    try:
        creds = _adc()
        client = bigquery.Client(project=project, credentials=creds)
       
        # Get table metadata
        table_ref = f"{project}.{dataset}.{table}"
        table_obj = client.get_table(table_ref)
       
        # Extract column information
        columns = []
        for field in table_obj.schema:
            columns.append({
                "name": field.name,
                "type": field.field_type,
                "mode": field.mode
            })
       
        # Extract partition information
        partition_info = None
        if table_obj.time_partitioning:
            partition_info = {
                "type": table_obj.time_partitioning.type_,
                "field": table_obj.time_partitioning.field
            }
        elif table_obj.range_partitioning:
            partition_info = {
                "type": "RANGE",
                "field": table_obj.range_partitioning.field
            }
       
        return {
            "columns": columns,
            "partition": partition_info,
            "table_ref": table_ref
        }
       
    except Exception as e:
        
        return None

def build_schema_context(sql: str) -> str:
    """
    Build schema context string for all tables in the query.
    """
    tables = extract_table_references(sql)
   
    if not tables:
        return "No tables detected in query."
   
    log("SCHEMA", f"Extracting schemas for {len(tables)} table(s)...")
   
    schema_parts = []
    for project, dataset, table in tables:
        schema = get_table_schema(project, dataset, table)
        if not schema:
            continue
       
        table_ref = schema["table_ref"]
        columns = schema["columns"]
        partition = schema["partition"]
       
        # Build schema description
        col_list = ", ".join([f"{c['name']} ({c['type']})" for c in columns[:SCHEMA_COLUMN_LIMIT]])
        if len(columns) > SCHEMA_COLUMN_LIMIT:
            col_list += f", ... and {len(columns) - SCHEMA_COLUMN_LIMIT} more columns"
       
        partition_info = ""
        if partition:
            if partition["field"]:
                partition_info = f"\n  Partitioned by: {partition['field']} ({partition['type']})"
            else:
                partition_info = f"\n  Partitioned by: _PARTITIONTIME (ingestion time)"
       
        schema_parts.append(f"Table: {table_ref}\n  Columns: {col_list}{partition_info}")
       
        log("SCHEMA", f"✓ Fetched schema for {table_ref}",
            cols=len(columns),
            partitioned="yes" if partition else "no")
   
    if not schema_parts:
        return "Could not fetch schemas for tables in query."
   
    return "\n\n".join(schema_parts)

# ============================================================================
# OPTIMIZATION POTENTIAL DETECTION
# ============================================================================

def has_optimization_potential(sql: str) -> Tuple[bool, List[str]]:
    """
    Check if a query has optimization potential.
    Returns (has_potential, list_of_reasons)
    """
    if not sql or not sql.strip():
        return False, []
   
    sql_upper = sql.upper()
    sql_normalized = " ".join(sql.split())
    reasons = []
   
    # 1. Check for SELECT * (most reliable optimization)
    if OPT_CHECK_SELECT_STAR:
        select_star_pattern = r'SELECT\s+\*'
        if re.search(select_star_pattern, sql_upper) and not re.search(r'SELECT\s+COUNT\s*\(\s*\*\s*\)', sql_upper):
            reasons.append("SELECT_STAR")
   
    # 2. Check for SELECT * in subqueries/CTEs (also reliable)
    if OPT_CHECK_SUBQUERY_STAR:
        if re.search(r'\(\s*SELECT\s+\*', sql_upper):
            reasons.append("SUBQUERY_SELECT_STAR")
        elif re.search(r'WITH\s+\w+\s+AS\s*\(\s*SELECT\s+\*', sql_upper):
            reasons.append("CTE_SELECT_STAR")
   
    # 3. Check for missing partition filters on partitioned tables
    if OPT_CHECK_PARTITION:
        tables = extract_table_references(sql)
        for project, dataset, table in tables:
            schema = get_table_schema(project, dataset, table)
            if schema and schema["partition"]:
                partition_field = schema["partition"].get("field", "_PARTITIONTIME")
                # Check if partition field is referenced in WHERE clause
                if partition_field not in sql_upper and "_PARTITION" not in sql_upper:
                    reasons.append("MISSING_PARTITION_FILTER")
                    break
   
    # 4. Check for ORDER BY without LIMIT (minor optimization)
    if OPT_CHECK_ORDER_BY:
        has_order_by = bool(re.search(r'\bORDER\s+BY\b', sql_upper))
        has_limit = bool(re.search(r'\bLIMIT\b', sql_upper))
        if has_order_by and not has_limit and len(reasons) > 0:
            reasons.append("ORDER_BY_NO_LIMIT")
   
    return len(reasons) > 0, reasons

def is_query_too_complex(sql: str) -> Tuple[bool, str]:
    """
    Check if query is too complex for safe optimization.
    Returns (is_too_complex, reason)
    """
    sql_upper = sql.upper()
   
    # Check query length
    if len(sql) > COMPLEXITY_MAX_LENGTH:
        return True, f"Query too long ({len(sql)} characters, max {COMPLEXITY_MAX_LENGTH})"
   
    # Check for set operations
    if not COMPLEXITY_ALLOW_EXCEPT and 'EXCEPT' in sql_upper:
        return True, "Contains EXCEPT operation"
    if not COMPLEXITY_ALLOW_INTERSECT and 'INTERSECT' in sql_upper:
        return True, "Contains INTERSECT operation"
    if not COMPLEXITY_ALLOW_UNION and 'UNION' in sql_upper:
        return True, "Contains UNION operation"
   
    # Check nesting level
    nesting_level = sql.count('(SELECT') + sql.count('( SELECT')
    if nesting_level > COMPLEXITY_MAX_NESTING:
        return True, f"Too deeply nested ({nesting_level} subquery levels, max {COMPLEXITY_MAX_NESTING})"
   
    # Check for FULL OUTER JOIN
    if not COMPLEXITY_ALLOW_FULL_OUTER and ('FULL OUTER JOIN' in sql_upper or 'FULL JOIN' in sql_upper):
        return True, "Contains FULL OUTER JOIN"
   
    # Check for window functions
    if not COMPLEXITY_ALLOW_QUALIFY and 'QUALIFY' in sql_upper:
        return True, "Contains QUALIFY (window function)"
   
    # Check for too many JOINs
    join_count = len(re.findall(r'\bJOIN\b', sql_upper))
    if join_count > COMPLEXITY_MAX_JOINS:
        return True, f"Too many JOINs ({join_count}, max {COMPLEXITY_MAX_JOINS})"
   
    # Check for multiple CTEs
    if not COMPLEXITY_ALLOW_MULTI_CTE and 'WITH ' in sql_upper and sql_upper.count('WITH ') > 1:
        return True, "Multiple CTEs detected"
   
    return False, ""

def determine_optimization_method(reasons: List[str], optimized_query: str, original_query: str) -> str:
    """
    Determine what optimization methods were applied based on reasons and query changes.
    """
    methods = []
   
    if "SELECT_STAR" in reasons or "SUBQUERY_SELECT_STAR" in reasons or "CTE_SELECT_STAR" in reasons:
        methods.append("Column pruning (replaced SELECT *)")
   
    if "MISSING_PARTITION_FILTER" in reasons and optimized_query:
        # Check if partition filter was added
        if "WHERE" in optimized_query.upper() and "WHERE" not in original_query.upper():
            methods.append("Added partition filter")
        elif optimized_query.upper().count("WHERE") > original_query.upper().count("WHERE"):
            methods.append("Added partition filter")
   
    if "ORDER_BY_NO_LIMIT" in reasons:
        if "LIMIT" in optimized_query.upper() and "LIMIT" not in original_query.upper():
            methods.append("Added LIMIT clause")
   
    if not methods:
        methods.append("Schema-aware optimization")
   
    return ", ".join(methods)

def infer_query_purpose(sql: str, user_email: str) -> str:
    """
    Infer the purpose of the query based on patterns.
    """
    sql_upper = sql.upper()
   
    # Check for scheduled queries
    if "scheduled_query" in user_email or "gserviceaccount" in user_email:
        return "Automated/Scheduled Query"
   
    # Check for analytics patterns
    if any(word in sql_upper for word in ["COUNT(", "SUM(", "AVG(", "GROUP BY"]):
        return "Analytics/Reporting"
   
    # Check for data pipeline patterns
    if any(word in sql_upper for word in ["INSERT", "CREATE", "UPDATE", "MERGE"]):
        return "Data Pipeline/ETL"
   
    # Check for exploration patterns
    if "LIMIT" in sql_upper:
        limit_match = re.search(r'LIMIT\s+(\d+)', sql_upper)
        if limit_match and int(limit_match.group(1)) <= 100:
            return "Data Exploration"
   
    # Check for ad-hoc analysis
    if "ORDER BY" in sql_upper or "WHERE" in sql_upper:
        return "Ad-hoc Analysis"
   
    return "General Query"

# ============================================================================
# BIGQUERY VALIDATION
# ============================================================================

def validate_sql_dryrun(sql:str)->dict:
    try:
        creds=_adc()
        client=bigquery.Client(project=QUERY_PROJECT_ID,credentials=creds)
        job=client.query(sql, job_config=bigquery.QueryJobConfig(dry_run=True))
        job.result()
        return {"valid":True,"bytes":job.total_bytes_processed,"cost":_cost_from_bytes(job.total_bytes_processed)}
    except Exception as e:
        return {"valid":False,"bytes":None,"cost":None,"error":str(e)}

# ============================================================================
# BIGQUERY RESULTS WRITER
# ============================================================================

def write_results_to_bigquery(results: List[Dict[str, Any]]):
    """
    Write optimization results to BigQuery table.
    """
    results_project = CFG.get("results", {}).get("project_id")
    results_table = CFG.get("results", {}).get("table")
   
    if not results_project or not results_table:
        log("BQ_WRITE", "Results table not configured in config.yaml - skipping BigQuery write")
        return
   
    try:
        creds = _adc()
        client = bigquery.Client(project=results_project, credentials=creds)
       
        table_ref = f"{results_project}.{results_table}"
       
        # Define schema for results table
        schema = [
            bigquery.SchemaField("job_id", "STRING"),
            bigquery.SchemaField("user_email", "STRING"),
            bigquery.SchemaField("selection_criteria", "STRING"),
            bigquery.SchemaField("run_timestamp", "TIMESTAMP"),
            bigquery.SchemaField("optimization_potential_reasons", "STRING"),
            bigquery.SchemaField("optimization_method", "STRING"),
            bigquery.SchemaField("query_purpose", "STRING"),
            bigquery.SchemaField("original_bytes", "INTEGER"),
            bigquery.SchemaField("original_gb", "FLOAT"),
            bigquery.SchemaField("optimized_bytes", "INTEGER"),
            bigquery.SchemaField("optimized_gb", "FLOAT"),
            bigquery.SchemaField("original_cost", "FLOAT"),
            bigquery.SchemaField("optimized_cost", "FLOAT"),
            bigquery.SchemaField("cost_savings", "FLOAT"),
            bigquery.SchemaField("execution_time_seconds", "INTEGER"),
            bigquery.SchemaField("optimization_successful", "BOOLEAN"),
            bigquery.SchemaField("error_message", "STRING"),
            bigquery.SchemaField("original_query", "STRING"),
            bigquery.SchemaField("optimized_query", "STRING"),
        ]
       
        # Create or get table
        try:
            table = client.get_table(table_ref)
            log("BQ_WRITE", f"Table exists: {table_ref}")
        except:
            table = bigquery.Table(table_ref, schema=schema)
            table = client.create_table(table)
            log("BQ_WRITE", f"Created table: {table_ref}")
       
        # Insert rows
        errors = client.insert_rows_json(table_ref, results)
       
        if errors:
            log("BQ_WRITE", f"Errors inserting rows: {errors}")
        else:
            log("BQ_WRITE", f"✓ Successfully wrote {len(results)} rows to {table_ref}")
           
    except Exception as e:
        log("BQ_WRITE", f"Failed to write to BigQuery: {str(e)}")

# ============================================================================
# GEMINI OPTIMIZATION WITH SCHEMA AWARENESS
# ============================================================================

def _gemini_optimize(sql:str)->Optional[str]:
    if not GEMINI_ENABLED:
        log("GEMINI", "Disabled - skipping")
        return None
    try:
        # Build schema context
        schema_context = build_schema_context(sql)
       
        creds=_adc()
        vertexai.init(project=VERTEX_PROJECT_ID, location=VERTEX_LOCATION, credentials=creds)
        model=GenerativeModel(GEMINI_MODEL)
        prompt=PROMPT_TEMPLATE.format(schema_context=schema_context, sql=sql)
        cfg=GenerationConfig(
            temperature=GEMINI_TEMPERATURE,
            top_p=GEMINI_TOP_P,
            max_output_tokens=GEMINI_MAX_TOKENS,
            response_mime_type="text/plain"
        )
       
        log("GEMINI", "Sending request to Gemini with schema context...")
        resp=model.generate_content(prompt,generation_config=cfg)
        text=(resp.text or "").strip()
       
        log("GEMINI", f"Response received, length: {len(text)} chars")
       
        if text.lower().startswith("```sql"): text=text[6:].strip()
        if text.lower().startswith("```"): text=text[3:].strip()
        if text.endswith("```"): text=text[:-3].strip()
       
        if not text:
            log("GEMINI", "Empty response")
            return None
           
        # Log first 200 chars of response for debugging
        log("GEMINI", f"Response preview: {text[:200]}...")
       
        return text
    except Exception as e:
        log("GEMINI","Fail",error=str(e))
        return None

# ============================================================================
# CORE OPTIMIZATION FUNCTION
# ============================================================================

def optimize_query(original_sql:str, original_bytes:int, original_cost:float, job_id:str=None, user_email:str=None)->Dict[str,Any]:
    run_ts=datetime.now(timezone.utc).isoformat()
    result={"run_timestamp":run_ts,"job_id":job_id,"user_email":user_email,
            "original_query":original_sql,"optimized_query":None,
            "original_bytes":original_bytes,"optimized_bytes":None,
            "original_gb":round(original_bytes/(1024**3), 4),"optimized_gb":None,
            "original_cost":_cost_from_bytes(original_bytes),"optimized_cost":None,
            "cost_savings":None,"optimization_successful":False,"error_message":None,
            "optimization_potential_reasons":None,
            "optimization_method":None,
            "query_purpose":infer_query_purpose(original_sql, user_email or "")}

    log("VALIDATE","Dry-running original query...")
    ov=validate_sql_dryrun(original_sql)
    if not ov["valid"]:
        result["error_message"]=f"Dry-run failed: {ov['error']}"
        log("ERROR", "Original query validation failed")
        return result
    result["original_bytes"]=ov["bytes"]; result["original_cost"]=ov["cost"]
    result["original_gb"]=round(ov["bytes"]/(1024**3), 4)
   
    log("INFO", f"Original query: {ov['bytes']:,} bytes, ${ov['cost']:.6f}")

    log("OPTIMIZE","Requesting Gemini optimization...")
    opt_sql=_gemini_optimize(original_sql)
   
    if not opt_sql:
        result["error_message"]="Gemini returned no optimization"
        log("ERROR", "Gemini returned None or empty string")
        return result
       
    # Check if query actually changed
    original_normalized = " ".join(original_sql.split())
    optimized_normalized = " ".join(opt_sql.split())
   
    if original_normalized == optimized_normalized:
        result["error_message"]="Gemini returned unchanged query"
        log("ERROR", "Gemini returned the same query (no changes)")
        return result

    log("INFO", f"Gemini made changes - query length: {len(original_sql)} -> {len(opt_sql)}")
   
    # Save debug queries
    if SAVE_DEBUG_QUERIES:
        debug_dir = CFG["output"].get("dir", "Optimise_SQL") + "/debug_queries"
        os.makedirs(debug_dir, exist_ok=True)
        with open(f"{debug_dir}/original_{job_id}.sql", "w") as f:
            f.write(original_sql)
        with open(f"{debug_dir}/optimized_{job_id}.sql", "w") as f:
            f.write(opt_sql)
        log("DEBUG", f"Saved queries to {debug_dir}/ for inspection")
   
    log("VALIDATE","Dry-running optimized query...")
    dv=validate_sql_dryrun(opt_sql)
    if not dv["valid"]:
        result["error_message"]=f"Optimized dry-run failed: {dv['error']}"
        log("ERROR", f"Optimized query validation failed: {dv['error'][:100]}")
        return result

    result["optimized_query"]=opt_sql
    result["optimized_bytes"]=dv["bytes"]
    result["optimized_gb"]=round(dv["bytes"]/(1024**3), 4)
    result["optimized_cost"]=dv["cost"]
    result["cost_savings"]=round(result["original_cost"]-result["optimized_cost"],COST_DECIMALS)
   
    log("INFO", f"Optimized query: {dv['bytes']:,} bytes, ${dv['cost']:.6f}")

    bytes_reduction=result["original_bytes"]-result["optimized_bytes"]
    if bytes_reduction>0:
        result["optimization_successful"]=True
        log("DECISION","✅ Success (any positive bytes reduction accepted)",
            reduction=bytes_reduction,orig=result["original_bytes"],opt=result["optimized_bytes"])
    elif bytes_reduction==0:
        result["error_message"]="No change in bytes"
        log("DECISION","⚠️ Neutral - no difference in bytes scanned")
    else:
        result["error_message"]=f"Optimization increased bytes by {abs(bytes_reduction)}"
        log("DECISION","❌ Worse (bytes increased)")

    return result

# ============================================================================
# QUERY SELECTION LOGIC
# ============================================================================

def get_top_queries_by_criteria(client: bigquery.Client, window_days: int = 7, limit_per_criteria: int = 10) -> List[Tuple[Any, str]]:
    """
    Fetch queries by bytes, cost, and execution time until we have the target number
    that meet optimization criteria (have potential AND not too complex).
    Returns list of tuples: (query_row, selection_criteria, reasons)
    Only includes unique queries with optimization potential.
    """
   
    base_sql = """
    SELECT
      job_id, user_email, creation_time, total_bytes_processed, total_bytes_billed,
      ROUND((total_bytes_billed / POW(10, 12)) * 5, 4) AS estimated_cost_usd,
      TIMESTAMP_DIFF(end_time, start_time, SECOND) AS execution_time_seconds,
      query
    FROM region-us.INFORMATION_SCHEMA.JOBS_BY_PROJECT
    WHERE
      state='DONE' AND job_type='QUERY'
      AND creation_time >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {days} DAY)
      AND query IS NOT NULL
      AND error_result IS NULL
      AND total_bytes_processed > 0
      AND statement_type = 'SELECT'
    """
   
    seen_job_ids: Set[str] = set()
    selected_queries: List[Tuple[Any, str]] = []
   
    print(f"\n{'='*80}")
    print(f"Searching for {limit_per_criteria} optimizable queries per criterion...")
    print(f"{'='*80}\n")
   
    fetch_limit = CFG["selection"].get("fetch_limit", 50)
   
    # Helper function to find queries meeting criteria
    def find_qualifying_queries(order_by: str, criteria_name: str) -> int:
        """Fetch queries ordered by criteria until we find enough that qualify"""
        sql = base_sql.format(days=window_days) + f"\nORDER BY {order_by}\nLIMIT {fetch_limit}"

        rows = list(client.query(sql).result())
        found_count = 0

        for row in rows:
            if row.job_id in seen_job_ids:
                continue

            is_complex, _ = is_query_too_complex(row.query)
            if is_complex:
                continue

            has_potential, reasons = has_optimization_potential(row.query)
            if not has_potential:
                continue

            # Query qualifies!
            seen_job_ids.add(row.job_id)
            selected_queries.append((row, criteria_name, reasons))
            found_count += 1

            if found_count >= limit_per_criteria:
                break

        # single concise line per criterion
        print(f"✓ Found {found_count}/{limit_per_criteria} qualifying queries for {criteria_name}")
        return found_count

   
    # Search for queries by each criterion
    find_qualifying_queries("total_bytes_processed DESC", "TOP_BYTES")
    find_qualifying_queries("estimated_cost_usd DESC", "TOP_COST")
    find_qualifying_queries("execution_time_seconds DESC", "TOP_TIME")
   
    return selected_queries

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    out_dir = CFG["output"].get("dir", "Optimise_SQL")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "optimization_results.csv")
   
    window_days = CFG.get("window", {}).get("days", 7)
    limit_per_criteria = CFG.get("selection", {}).get("limit_per_criteria", 10)

    client = bigquery.Client(project=QUERY_PROJECT_ID)
   
    # Get queries using the multi-criteria selection
    selected_queries = get_top_queries_by_criteria(client, window_days, limit_per_criteria)
   
    print(f"\n{'='*80}")
    print(f"Selected {len(selected_queries)} unique queries with optimization potential")
    print(f"{'='*80}\n")
   
    # Collect all results for BigQuery insert
    all_results = []
   
    # Process queries with optimization potential
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "job_id","user_email","selection_criteria","run_timestamp","optimization_potential_reasons",
            "optimization_method","query_purpose","original_bytes","original_gb","optimized_bytes","optimized_gb",
            "original_cost","optimized_cost","cost_savings","execution_time_seconds",
            "optimization_successful","error_message"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i, (q, criteria, reasons) in enumerate(selected_queries, 1):
            print("\n" + "="*80)
            print(f"Query {i}/{len(selected_queries)} | {q.job_id} | {q.user_email}")
            print(f"Selection: {criteria} | Potential: {', '.join(reasons)}")
            print("="*80)

            result = optimize_query(
                original_sql=q.query,
                original_bytes=q.total_bytes_processed,
                original_cost=q.estimated_cost_usd,
                job_id=q.job_id,
                user_email=q.user_email,
            )
           
            result["optimization_potential_reasons"] = ", ".join(reasons)
            result["selection_criteria"] = criteria
            result["execution_time_seconds"] = q.execution_time_seconds
           
            # Determine optimization method
            result["optimization_method"] = determine_optimization_method(
                reasons,
                result.get("optimized_query"),
                q.query
            )

            # Write to CSV
            writer.writerow({
                "job_id": result.get("job_id"),
                "user_email": result.get("user_email"),
                "selection_criteria": criteria,
                "run_timestamp": datetime.now(timezone.utc).isoformat(),
                "optimization_potential_reasons": result.get("optimization_potential_reasons"),
                "optimization_method": result.get("optimization_method"),
                "query_purpose": result.get("query_purpose"),
                "original_bytes": result.get("original_bytes"),
                "original_gb": result.get("original_gb"),
                "optimized_bytes": result.get("optimized_bytes"),
                "optimized_gb": result.get("optimized_gb"),
                "original_cost": result.get("original_cost"),
                "optimized_cost": result.get("optimized_cost"),
                "cost_savings": result.get("cost_savings"),
                "execution_time_seconds": q.execution_time_seconds,
                "optimization_successful": result.get("optimization_successful"),
                "error_message": result.get("error_message"),
            })
            f.flush()
           
            # Collect for BigQuery insert
            all_results.append(result)

    print(f"\n{'='*80}")
    print(f"Results written to: {csv_path}")
    print(f"{'='*80}")
   
    # Write all results to BigQuery
    if all_results:
        print(f"\n{'='*80}")
        print(f"Writing {len(all_results)} results to BigQuery...")
        print(f"{'='*80}\n")
        write_results_to_bigquery(all_results)

if __name__ == "__main__":
    main()
