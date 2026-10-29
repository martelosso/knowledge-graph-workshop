from __future__ import annotations

import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from rdflib import Graph

ROOT                  = Path(__file__).parent.parent
DEFAULT_ONTOLOGY_PATH = ROOT / "ontology" / "ontology.owl.ttl"
DEFAULT_DATA_PATH     = ROOT / "data" / "mock_data.ttl"
DEFAULT_MODEL         = os.environ.get("OPENROUTER_MODEL", "anthropic/claude-5.5-sonnet")
OPENROUTER_BASE_URL   = "https://openrouter.ai/api/v1"

SPARQL_PREFIXES = """\
PREFIX :     <https://example.org/smarket/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX owl:  <http://www.w3.org/2002/07/owl#>
PREFIX xsd:  <http://www.w3.org/2001/XMLSchema#>\
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "sparql_query",
            "description": (
                "Run a SPARQL SELECT query against the knowledge graph and return rows. "
                "Always include PREFIX declarations at the top. "
                "If you get an error or empty results, inspect the ontology and refine the query."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A valid SPARQL SELECT query including PREFIX declarations.",
                    }
                },
                "required": ["query"],
            },
        },
    }
]

SYSTEM_PROMPT_TEMPLATE = """\
You are a retail data assistant for Smarket, a Colombian supermarket chain.

Your context contains ONLY the ontology schema — the vocabulary of classes and \
properties that describe the data. All actual records (stores, items, prices, \
promotions, customers, orders) live in a separate knowledge graph that you must \
query with the `sparql_query` tool.

How to answer factual questions:
  1. Read the ontology to identify the relevant classes and properties.
  2. Write a SPARQL SELECT query using those classes and properties.
  3. Call `sparql_query`. If it returns an error or empty results, revise and retry.
  4. Use the returned rows to write a clear, direct answer.

How to answer schema questions ("What classes exist?", "How is pricing modelled?"):
  Answer directly from the ontology — no tool call needed.

Always include these prefixes in every query:
{prefixes}

━━━ ONTOLOGY (schema only — no instance data here) ━━━━━━━━━━━━━━━━━━━━━━━━━

```turtle
{ontology}
```
"""

# Cache the parsed data graph so we don't re-parse on every question.
_data_graph: Graph | None = None


def _get_data_graph(data_path: Path) -> Graph:
    global _data_graph
    if _data_graph is None:
        _data_graph = Graph()
        _data_graph.parse(str(data_path), format="turtle")
    return _data_graph


def _describe_traversal(patterns: list, count: int, labels: list) -> list:
    """Turn triple patterns into plain-language Spanish steps."""
    steps = []
    seen_label_step = False

    for s_tok, p_tok, o_tok in patterns:
        p_local = re.search(r"[#/:]([^#/:]+)$", p_tok)
        p_name  = p_local.group(1) if p_local else p_tok

        is_type    = p_tok in ("a",) or p_tok.endswith(":type") or p_tok.endswith("#type")
        is_literal = o_tok.startswith('"')
        is_label   = "label" in p_name.lower()

        if is_type:
            o_local = re.search(r"[#/:]([^#/:]+)$", o_tok)
            steps.append(f"Buscó todos los nodos de tipo **{o_local.group(1) if o_local else o_tok}**")
        elif is_literal:
            lit = o_tok.strip('"').split('"')[0]
            steps.append(f"Filtró donde **{p_name}** = \"{lit}\"")
        elif is_label and not seen_label_step:
            steps.append("Obtuvo las etiquetas de cada nodo")
            seen_label_step = True
        else:
            steps.append(f"Siguió la propiedad **{p_name}** en el grafo")

    # Final result summary
    if labels:
        sample = ", ".join(labels[:4])
        extra  = f" y {count - len(labels[:4])} más" if count > 4 else ""
        steps.append(f"Encontró **{count}** resultado{'s' if count != 1 else ''}: {sample}{extra}")
    else:
        steps.append("No encontró resultados")

    return steps


def _extract_schema_used(query: str) -> dict:
    """Scan the raw SPARQL query text for all smarket-namespace IRIs.
    Works regardless of whether triples use ; shorthand, inline syntax, etc."""
    SMARKET = "https://example.org/smarket/"
    iris: set[str] = set()

    # :localName — colon-prefix shorthand; lookbehind skips rdfs:, owl:, prefix:local, etc.
    for m in re.finditer(r'(?<![A-Za-z0-9_<#/]):([A-Za-z][A-Za-z0-9_-]*)', query):
        iris.add(SMARKET + m.group(1))

    # <https://example.org/smarket/localName>
    for m in re.finditer(r'<https://example\.org/smarket/([^>]+)>', query):
        iris.add(SMARKET + m.group(1))

    return {"iris": list(iris)}


def _build_traversal(query: str, columns: list, rows: list, g: Graph) -> list:
    """
    Parse the SPARQL WHERE clause and resolve each triple pattern against the
    actual bound values, building a list of graph edges (from, prop, to) that
    represent the path the agent walked through the knowledge graph.
    """
    from rdflib import URIRef
    from rdflib.namespace import RDFS, RDF

    SMARKET = "https://example.org/smarket/"

    PREFIXES = {
        ":":     SMARKET,
        "rdfs:": str(RDFS),
        "rdf:":  str(RDF),
        "owl:":  "http://www.w3.org/2002/07/owl#",
        "xsd:":  "http://www.w3.org/2001/XMLSchema#",
    }

    def expand(tok: str) -> str:
        tok = tok.strip()
        if tok == "a":
            return str(RDF.type)
        if tok.startswith("<") and tok.endswith(">"):
            return tok[1:-1]
        for pfx, ns in PREFIXES.items():
            if tok.startswith(pfx):
                return ns + tok[len(pfx):]
        return tok

    def label_of(uri: str) -> str:
        for lbl in g.objects(URIRef(uri), RDFS.label):
            return str(lbl)
        m = re.search(r"[#/]([^#/]+)$", uri)
        return m.group(1) if m else uri

    def short(uri: str) -> str:
        if uri.startswith(SMARKET):
            return ":" + uri[len(SMARKET):]
        m = re.search(r"[#/]([^#/]+)$", uri)
        return m.group(1) if m else uri

    def is_uri(s: str) -> bool:
        return s.startswith("http")

    def display(val: str) -> str:
        if is_uri(val):
            lbl = label_of(val)
            sh  = short(val)
            return f"{lbl} ({sh})" if lbl and lbl != sh.lstrip(":") else sh
        return val.strip('"').split('"')[0]  # bare literal text

    # Extract WHERE clause body
    where = re.search(
        r"WHERE\s*\{(.+)\}\s*(?:ORDER|LIMIT|GROUP|HAVING|$)",
        query, re.DOTALL | re.IGNORECASE,
    )
    if not where:
        return []
    body = where.group(1)

    # Match triple patterns (handles ?var, :local, prefix:local, <IRI>, "string"@lang)
    TERM = (
        r'(?:\?[\w]+|<[^>]*>|[\w-]*:[\w.-]+|"[^"]*"'
        r'(?:[@^][^\s.;,{}\(\)]*)?|true|false|-?\d+)'
    )
    triples = re.findall(
        rf"({TERM})\s+({TERM})\s+({TERM})\s*[.;,]?", body, re.IGNORECASE
    )

    seen_edges: set = set()
    all_edges: list = []

    for row in rows[:12]:
        row_dict = dict(zip(columns, row))

        for s_tok, p_tok, o_tok in triples:
            # Resolve ?variable references to their bound values
            s = row_dict.get(s_tok.lstrip("?")) if s_tok.startswith("?") else expand(s_tok)
            p = expand(p_tok)
            o = (
                row_dict.get(o_tok.lstrip("?"))
                if o_tok.startswith("?")
                else (o_tok.strip('"').split('"')[0] if o_tok.startswith('"') else expand(o_tok))
            )

            if not s or not p or not o:
                continue

            edge_key = (s, p, o)
            if edge_key in seen_edges:
                continue
            seen_edges.add(edge_key)

            all_edges.append({
                "from":    display(s),
                "from_id": short(s) if is_uri(s) else s,
                "prop":    short(p),
                "to":      display(o),
                "to_id":   short(o) if is_uri(o) else o,
            })

    # Group edges by their source node → list of node objects with outgoing edges
    from collections import defaultdict
    node_edges: dict = defaultdict(list)
    node_labels: dict = {}
    for edge in all_edges:
        fid = edge["from_id"]
        node_edges[fid].append({"prop": edge["prop"], "to": edge["to"], "to_id": edge["to_id"]})
        node_labels[fid] = edge["from"]

    # Find root nodes: appear as "from" but are never the "to" of another edge
    all_to_ids = {e["to_id"] for e in all_edges}
    root_ids   = [fid for fid in node_edges if fid not in all_to_ids]

    # Fall back: if every node is a "to" of something else, use all of them
    if not root_ids:
        root_ids = list(node_edges.keys())

    return [
        {"id": rid, "label": node_labels[rid], "edges": node_edges[rid]}
        for rid in root_ids
    ]


def execute_sparql(query: str, data_path: Path = DEFAULT_DATA_PATH) -> dict:
    """Run a SPARQL SELECT against the data graph. Returns {ok, traversal, count} or {ok, error}."""
    try:
        g = _get_data_graph(data_path)
        result = g.query(query)
        columns = [str(v) for v in result.vars]
        rows    = [
            [str(cell) if cell is not None else "" for cell in row]
            for row in result
        ]
        traversal = _build_traversal(query, columns, rows, g)
        # Visual metadata for the UI
        where = re.search(r"WHERE\s*\{(.+)\}", query, re.DOTALL | re.IGNORECASE)
        TERM = (r'(?:\?[\w]+|<[^>]*>|[\w-]*:[\w.-]+|"[^"]*"'
                r'(?:[@^][^\s.;,{}\(\)]*)?|true|false|-?\d+)')
        patterns = re.findall(rf"({TERM})\s+({TERM})\s+({TERM})\s*[.;,]?",
                              where.group(1) if where else "", re.IGNORECASE)
        root_labels = [n["label"].split(" (")[0] for n in traversal[:4]]
        description = _describe_traversal(patterns, len(rows), root_labels)
        schema_used = _extract_schema_used(query)  # scan raw query text, handles all SPARQL syntax
        return {
            # ── sent to the model so it can reason over the data ──
            "ok":      True,
            "columns": columns,
            "rows":    rows,
            "count":   len(rows),
            # ── visual metadata used only by the UI ──
            "traversal":   traversal,
            "description": description,
            "schema_used": schema_used,
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def build_system_prompt(ontology_path: Path = DEFAULT_ONTOLOGY_PATH) -> str:
    return SYSTEM_PROMPT_TEMPLATE.format(
        prefixes=SPARQL_PREFIXES,
        ontology=ontology_path.read_text(),
    )


def build_client() -> OpenAI:
    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Add it to your .env file or export it."
        )
    return OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)


def ask(
    messages: list[dict],
    system_prompt: str,
    client: OpenAI,
    model: str = DEFAULT_MODEL,
    data_path: Path = DEFAULT_DATA_PATH,
    max_iterations: int = 6,
) -> dict:
    """
    Run the agentic loop. Returns:
      {
        "answer": str,
        "steps": [
          {"type": "sparql",  "content": "<query string>"},
          {"type": "results", "content": {ok, columns, rows, count}},
          {"type": "error",   "content": {ok, error}},
        ]
      }
    """
    steps: list[dict] = []
    msgs = [{"role": "system", "content": system_prompt}] + messages

    for _ in range(max_iterations):
        response = client.chat.completions.create(
            model=model,
            messages=msgs,
            tools=TOOLS,
            tool_choice="auto",
        )

        choice = response.choices[0]
        msg    = choice.message

        if msg.tool_calls:
            # Append assistant turn to conversation
            msgs.append({
                "role":       "assistant",
                "content":    msg.content or "",
                "tool_calls": [
                    {
                        "id":   tc.id,
                        "type": "function",
                        "function": {
                            "name":      tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ],
            })

            # Execute each tool call
            for tc in msg.tool_calls:
                if tc.function.name != "sparql_query":
                    continue
                args  = json.loads(tc.function.arguments)
                query = args.get("query", "")

                steps.append({"type": "sparql", "content": query})
                result = execute_sparql(query, data_path)

                # What goes to the UI (visual traversal, no raw rows)
                ui_payload = {k: result[k] for k in
                              ("ok", "traversal", "description", "count", "schema_used", "error")
                              if k in result}
                steps.append({"type": "results" if result["ok"] else "error", "content": ui_payload})

                # What goes to the model (actual data so it can answer)
                model_payload = {k: result[k] for k in
                                 ("ok", "columns", "rows", "count", "error")
                                 if k in result}
                msgs.append({
                    "role":         "tool",
                    "tool_call_id": tc.id,
                    "content":      json.dumps(model_payload),
                })
        else:
            # Final answer — no more tool calls
            return {"answer": msg.content or "", "steps": steps}

    return {"answer": "No pude completar la consulta en el número de pasos permitidos.", "steps": steps}


def main() -> None:
    import sys

    system_prompt = build_system_prompt()
    client        = build_client()

    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        result = ask([{"role": "user", "content": question}], system_prompt, client)
        print(result["answer"])
        for step in result["steps"]:
            if step["type"] == "sparql":
                print(f"\n[SPARQL]\n{step['content']}")
            elif step["type"] == "results":
                c = step["content"]
                print(f"\n[Results: {c.get('count', 0)} rows] {c.get('columns', [])}")
                for row in c.get("rows", [])[:5]:
                    print(" ", row)
            elif step["type"] == "error":
                print(f"\n[Error] {step['content'].get('error')}")
        return

    print("Smarket assistant ready (multi-turn). Ctrl-D to exit.\n")
    history: list[dict] = []
    while True:
        try:
            question = input("You: ").strip()
        except EOFError:
            print()
            break
        if not question:
            continue
        history.append({"role": "user", "content": question})
        result = ask(history, system_prompt, client)
        answer = result["answer"]
        history.append({"role": "assistant", "content": answer})
        print(f"\nAssistant: {answer}\n")


if __name__ == "__main__":
    main()
