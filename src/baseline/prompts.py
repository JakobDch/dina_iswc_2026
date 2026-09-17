from langchain_core.prompts import ChatPromptTemplate


OUTPUT_SEPARATOR = "### ANSWER ###"


# --- FEW-SHOT EXAMPLES FOR EACH PIPELINE STEP ---
FEW_SHOT_EXAMPLES = {
    "IdentifyKeywords": '''
Here is an example of how to solve the task:
--- Start Example 1 ---
User query: "Show me all construction sites in Wuppertal and Düsseldorf that have May as their start month."
Answer:
{
  "search_terms": [
    "construction site Wuppertal",
    "construction site Düsseldorf",
    "construction site start month"
  ]
}
--- End Example 1 ---

--- Start Example 2 ---
User query: "Give me all suspended railway stops in Wuppertal that contain 'Straße' or 'Strasse' in their name"
Answer:
{
  "search_terms": [
    "suspended railway stop Wuppertal",
    "suspended railway stop name"
  ]
}
--- End Example 2 ---

--- Start Example 3 ---
User query: "Overview of stop signs in Hamburg, including mounting location and direction."
Answer:
{
  "search_terms": [
    "stop sign Hamburg",
    "stop sign mounting location",
    "stop sign direction"
  ]
}
--- End Example 3 ---

''',
    "SelectAndValidateModels": '''
Here is an example of how to solve the task:

--- Start Example 1 ---
User query: "Overview of traffic light systems in Hamburg, including mounting location and direction."
Triples of semantic models identified as relevant:

Model: 003.ttl
```
local:Fahrtrichtungspfeil local:ausrichtung local:Gradma_ .
local:Fahrtrichtungspfeil local:befindet_sich_an local:Geographisches_Polygon .
local:Fahrtrichtungspfeil local:hat local:Fahrtrichtungspfeiltyp .
local:Fahrtrichtungspfeil local:located_in "Rostock"^^xsd:string .
local:Geographischer_Punkt local:besteht_aus local:UTM_false_easting .
local:Geographischer_Punkt local:besteht_aus local:UTM_northing .
local:Geographisches_Polygon local:besteht_aus local:Geographischer_Punkt .
local:Geographisches_Polygon local:koordinatenreferenzsystem xsd:string .
local:Gradma_ local:absolute_Bezugsrichtung local:Geographischer_Norden .
```

Model: 020.ttl
```
local:CarSharing_Station local:befindet_sich_an local:Geographischer_Punkt .
local:CarSharing_Station local:betrieben_durch local:Cambio .
local:CarSharing_Station local:bietet_an local:Fahrzeug .
local:CarSharing_Station local:hat local:Adresse .
local:CarSharing_Station local:hat local:Bezeichnung .
local:CarSharing_Station local:hat local:Standortbezeichnung .
local:CarSharing_Station local:hat local:Zugangsinformation .
local:CarSharing_Station local:hat local:Zusatzinformation .
local:CarSharing_Station local:identifiziert_durch local:Identifikator .
local:CarSharing_Station local:located_in "Berlin"^^xsd:string .
local:Adresse local:besteht_aus local:Hausnummer .
local:Adresse local:besteht_aus local:Land .
local:Adresse local:besteht_aus local:Postleitzahl .
local:Adresse local:besteht_aus local:Stadt .
local:Adresse local:besteht_aus local:Stra_e .
local:Fahrzeug local:geh_rt_zu local:Kategorie .
local:Geographischer_Punkt local:besteht_aus local:Geographische_Breite .
local:Geographischer_Punkt local:besteht_aus local:Geographische_L_nge .
local:Kategorie local:identifiziert_durch local:Identifikator .
```

Model: 078.ttl
```
local:Lichtsignalanlage local:befindet_sich_an local:Geographischer_Punkt .
local:Lichtsignalanlage local:erfasst_am local:Zeitstempel .
local:Lichtsignalanlage local:hat local:Befestigungsort .
local:Lichtsignalanlage local:hat local:H_he .
local:Lichtsignalanlage local:hat local:Richtung .
local:Lichtsignalanlage local:hat local:Verwendungszweck .
local:Lichtsignalanlage local:identifiziert_durch local:Identifikator .
local:Lichtsignalanlage local:identifiziert_durch local:Lichtsignalanlagen_Nummer .
local:Lichtsignalanlage local:located_in "Hamburg"^^xsd:string .
local:Befestigungsort local:identifiziert_durch local:Identifikator .
local:Ellipsoidische_H_he local:hat local:Standardabweichung .
local:Geographischer_Punkt local:besteht_aus local:Ellipsoidische_H_he .
local:Geographischer_Punkt local:besteht_aus local:UTM_false_easting .
local:Geographischer_Punkt local:besteht_aus local:UTM_northing .
local:Geographischer_Punkt local:koordinatenreferenzsystem xsd:string .
local:Richtung local:ausrichtung local:Geographischer_Norden .
local:Richtung local:gemessen_in local:Gradma_ .
local:UTM_false_easting local:hat local:Standardabweichung .
local:UTM_northing local:hat local:Standardabweichung .
local:Zeitstempel local:zeitstempelformat xsd:string .
```

Answer:
{
  "evaluation": "fully_possible",
  "reasoning": "Model 078.ttl contains the class 'local:Lichtsignalanlage' with the required properties 'local:hat local:Befestigungsort' and 'local:hat local:Richtung' as well as the geographic assignment to Hamburg via 'local:located_in Hamburg'. All aspects of the user query (overview of traffic light systems in Hamburg including mounting location and direction) can be fully covered. The other models (003.ttl for directional arrows in Rostock and 020.ttl for car sharing stations in Berlin) are not relevant for this query."
}

--- End Example 1 ---

--- Start Example 2 ---

User query: "Give me all suspended railway stops in Wuppertal including construction year"
Triples of semantic models identified as relevant:

Model: 061.ttl
```
local:Stra_enbeleuchtung local:befindet_sich_an local:Geographischer_Punkt .
local:Stra_enbeleuchtung local:identifiziert_durch local:Identifikator .
local:Stra_enbeleuchtung local:identifiziert_durch local:Standortbezeichnung .
local:Stra_enbeleuchtung local:located_in "Duisburg"^^xsd:string .
local:Geographischer_Punkt local:besteht_aus local:UTM_false_easting .
local:Geographischer_Punkt local:besteht_aus local:UTM_northing .
local:Geographischer_Punkt local:koordinatenreferenzsystem xsd:string .
```




Model: 099.ttl
```
local:Schwebebahnstation local:angefahren_von local:Wuppertaler_Schwebebahn .
local:Schwebebahnstation local:befindet_sich_an local:Geographischer_Punkt .
local:Schwebebahnstation local:hat local:Bezeichnung .
local:Schwebebahnstation local:ist_ein_e_ local:Haltestelle .
local:Schwebebahnstation local:located_in "Wuppertal"^^xsd:string .
local:Geographischer_Punkt local:besteht_aus local:Geographische_Breite .
local:Geographischer_Punkt local:besteht_aus local:Geographische_L_nge .
local:Geographischer_Punkt local:koordinatenreferenzsystem xsd:string .
```



Model: 064.ttl
```
local:Haltestelle local:benutzt_von local:Zug .
local:Haltestelle local:hat local:Bezeichnung .
local:Haltestelle local:hat local:Z_hlung .
local:Haltestelle local:identifiziert_durch local:Haltestellennummer .
local:Haltestelle local:located_in "Berlin"^^xsd:string .
local:Aussteiger local:ist_ein_e_ local:Mensch .
local:Einsteiger local:ist_ein_e_ local:Mensch .
local:Tag local:befindet_sich_in local:Jahr .
local:Tag local:durchschnitt local:Aussteiger .
local:Tag local:durchschnitt local:Einsteiger .
local:Tag local:ist_vom_Typ local:Tagestyp .
local:Z_hlung local:bezieht_sich_auf local:Tag .
```

Answer:
{
  "evaluation": "partially_possible",
  "reasoning": "Model 099.ttl contains the class 'local:Schwebebahnstation' with the required property 'local:hat local:Bezeichnung' as well as the geographic assignment to Wuppertal via 'local:located_in Wuppertal'. The suspended railway stops can be identified and queried. However, the information about the construction year of the stops is missing in the semantic model, which means this aspect of the query cannot be covered.",
  "user_clarification": "The semantic model does not contain information about the construction year of the suspended railway stops. Therefore, the construction year cannot be considered in the query. Should the SPARQL query still be created to query all suspended railway stops in Wuppertal (without the construction year information)?"
}

--- End Example 2 ---
''',
    "SPARQLGeneration": '''
Here are some examples of how to solve the task:
--- Start Example 1 ---
User query: "Give me all suspended railway stops in Wuppertal that contain 'Straße' or 'Strasse' in their name"
Semantic models:

Model: 099.ttl
```
local:Schwebebahnstation local:angefahren_von local:Wuppertaler_Schwebebahn .
local:Schwebebahnstation local:befindet_sich_an local:Geographischer_Punkt .
local:Schwebebahnstation local:hat local:Bezeichnung .
local:Schwebebahnstation local:ist_ein_e_ local:Haltestelle .
local:Schwebebahnstation local:located_in "Wuppertal"^^xsd:string .
local:Geographischer_Punkt local:besteht_aus local:Geographische_Breite .
local:Geographischer_Punkt local:besteht_aus local:Geographische_L_nge .
local:Geographischer_Punkt local:koordinatenreferenzsystem xsd:string .
```

Notes from the model check: The model is suitable.
Answer:
PREFIX owl:    <http://www.w3.org/2002/07/owl#>
PREFIX plasma: <http://plasma.uni-wuppertal.de/ontology#>
PREFIX plcm:   <http://plasma.uni-wuppertal.de/cm#>
PREFIX plsm:   <http://plasma.uni-wuppertal.de/sm/>
PREFIX rdf:    <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs:   <http://www.w3.org/2000/01/rdf-schema#>
PREFIX xsd:    <http://www.w3.org/2001/XMLSchema#>

SELECT DISTINCT ?Name
WHERE {{
  ?station a local:Schwebebahnstation ;
        local:hat ?Bezeichnung ;
        local:located_in "Rostock"^^xsd:string .  
  ?Bezeichnung plasma:hasValue ?RoherWert .
  BIND(LCASE(?RoherWert) AS ?NameLower)
  FILTER (CONTAINS(?NameLower, "straße") || CONTAINS(?NameLower, "strasse"))
  BIND(?RoherWert AS ?Name)
}}
--- End Example 1 ---



--- Start Example 2 ---
User query: "Give me all park-and-ride facilities in Rostock with at least 10 parking spaces including location designation. Return the instance, the number of parking spaces, and the location designation (total of 3 columns)."

Semantic models:

Model: 033.ttl
```
local:Park_and_Ride_Anlage local:anzahl local:Stellplatz .
local:Park_and_Ride_Anlage local:befindet_sich_an local:Geographischer_Punkt .
local:Park_and_Ride_Anlage local:hat local:Adresse .
local:Park_and_Ride_Anlage local:hat local:Anbindung .
local:Park_and_Ride_Anlage local:hat local:Bezeichnung .
local:Park_and_Ride_Anlage local:hat local:Standortbezeichnung .
local:Park_and_Ride_Anlage local:hat local:Zusatzinformation .
local:Park_and_Ride_Anlage local:hat local:_ffnungszeit .
local:Park_and_Ride_Anlage local:identifiziert_durch local:Identifikator .
local:Park_and_Ride_Anlage local:located_in "Rostock"^^xsd:string .

local:Adresse local:besteht_aus local:Postleitzahl .
local:Adresse local:besteht_aus local:Stadt .

local:Geographischer_Punkt local:besteht_aus local:Geographische_Breite .
local:Geographischer_Punkt local:besteht_aus local:Geographische_L_nge .
local:Geographischer_Punkt local:koordinatenreferenzsystem xsd:string .

local:Website local:ist_vom_Typ local:Uniform_Resource_Locator__URL_ .

local:Zusatzinformation local:dargestellt_auf local:Website .


Notes from the model check: The model is suitable.

Answer:

PREFIX owl:    <http://www.w3.org/2002/07/owl#>
PREFIX plasma: <http://plasma.uni-wuppertal.de/ontology#>
PREFIX plcm:   <http://plasma.uni-wuppertal.de/cm#>
PREFIX plsm:   <http://plasma.uni-wuppertal.de/sm/>
PREFIX rdf:    <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs:   <http://www.w3.org/2000/01/rdf-schema#>
PREFIX xsd:    <http://www.w3.org/2001/XMLSchema#>
PREFIX local:  <https://local.ontology#>

SELECT DISTINCT
  ?anlage
  (xsd:integer(?stellplatzWert) AS ?Anzahl_Stellplaetze)
  ?Standortbezeichnung
WHERE {
  ?anlage a local:Park_and_Ride_Anlage ;
          local:located_in "Rostock"^^xsd:string ;
          local:anzahl ?stellplatzEntity ;
          local:hat ?standortbezEntity .

  # Get parking space count as value (filter by type)
  ?stellplatzEntity a local:Stellplatz ;
                    plasma:hasValue ?stellplatzWert .

  # Get location name as value (filter by type)
  ?standortbezEntity a local:Standortbezeichnung ;
                     plasma:hasValue ?Standortbezeichnung .

  FILTER( xsd:integer(?stellplatzWert) >= 10 )
}
```
--- End Example 2 ---
'''
}


# --- FEW-SHOT EXAMPLES FOR ONTOP WORKSPACES (Energy Domain Dataset Dataset) ---
FEW_SHOT_EXAMPLES_SPARQL_ONLY = {
    "SPARQLGeneration": '''
Here are some examples of how to solve the task:
--- Start Example 1 ---
User query: "List all unique pairs of member and core drilling where the core drilling is assigned to a wellbore and the total core length exceeds 300 meters. If the length is in feet, convert it to meters using 0.3048 first. Output the normalized length in meters together with member and core drilling."
Semantic models:

Model: schema_wellbores.ttl
```
eno:WellboreCoreSet eno:member eno:WellboreCore .
eno:WellboreCore eno:coreForWellbore eno:Wellbore .
eno:WellboreCore eno:coreForWellbore eno:AppraisalWellbore .
eno:WellboreCore eno:coreForWellbore eno:BlowoutWellbore .
eno:WellboreCore eno:coreForWellbore eno:DevelopmentWellbore .
eno:WellboreCore eno:coreForWellbore eno:ExplorationWellbore .
eno:WellboreCore eno:coresTotalLength xsd:decimal .
eno:WellboreCore eno:coreIntervalUOM xsd:string .
eno:WellboreCore eno:coreNo xsd:integer .
eno:WellboreCore eno:dateSynchronized xsd:date .
```

Notes from the model check: The model is suitable.
Answer:
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
PREFIX owl: <http://www.w3.org/2002/07/owl#>
PREFIX eno: <http://example.org/ontology/energy#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT DISTINCT ?member ?wc (?length AS ?lenghtM)
WHERE {
  ?member eno:member ?wc.
  ?wc eno:coreForWellbore [ rdf:type eno:Wellbore ] .
  {
    {
      ?wc eno:coresTotalLength ?l ;
      eno:coreIntervalUOM "[m   ]"^^xsd:string .
      BIND(?l AS ?length)
    }
    UNION
    {
      ?wc eno:coresTotalLength ?l ;
      eno:coreIntervalUOM "[ft   ]"^^xsd:string .
      BIND((?l * 0.3048) AS ?length)
    }
  }
  FILTER(?length > 300)
}
--- End Example 1 ---
'''
}





# Output Instructions for current pipeline steps

# STEP 1: Identify keywords
OUTPUT_INSTRUCTIONS_IDENTIFY_KEYWORDS = (
    "Provide your answer exclusively in the following JSON format - without greeting, explanation, or any other text:\n"
    "{\n  \"search_terms\": [ \"Suchbegriff1\", \"Suchbegriff2\", \"Suchbegriff3\"]\n}"
)



IDENTIFY_KEYWORDS_PROMPT_TEMPLATE = ChatPromptTemplate.from_template("""
This is the first sub-step in a process that converts a natural language user query into a SPARQL query.
The system is based on datasets and associated semantic models.
The semantic models have been converted into embeddings. Each triple (subject-predicate-object) is transformed into a numerical vector (embedding) and stored in a FAISS vector database. This index is the basis for semantic search.

Your task is to extract COMBINED SEARCH TERMS from the user query that are optimal for semantic search in the embedding.

IMPORTANT: Do NOT extract individual keywords, but meaningful COMBINATIONS of terms from the user query that together form a specific search context.
IMPORTANT: Think carefully about what keywords are relevant for the search terms to find relevant datasets. And what, for example, are instructions or requirements regarding the query, which are answered with SPARQL via FILTER, LIMIT or other operators like DESC and therefore irrelevant for the search after the needed semantic models.
IMPORTANT: Extract the search terms in THE SAME LANGUAGE as the user query. If the user query is in English, provide English search terms. If it's in German, provide German search terms.

Step 1: Extract the searched for data information from the user query. This should answer the question: What data is the user looking for?
Step 2: Identify any restrictions or conditions that limit the search results. This should answer the question: Under what conditions or restrictions should the data be retrieved? If this condition is too specific (e.g., only from a specific year, only from a specific city or person), do NOT include it in the search terms, as these conditions can usually be applied later in the SPARQL query via FILTER.
Step 3: Combine the identified data information and conditions into meaningful search terms that can be used for semantic search in the embedding index of the semantic models.


Guidelines:
1. Avoid individual, too general terms
2. Stay precise with the terms from the user query
3. Keep in mind that the search terms must cover all searched concepts for the corresponding query.
4. Convert plural to singular, if applicable
5. Convert umlauts to the correct letters, e.g., "Ä" becomes "ae" and "Ö" becomes "oe"
6. Ignore stop words and filler words
7. Each search term should comprise 2-4 words that belong together semantically
8. MAINTAIN THE LANGUAGE of the user query in your search terms

Here are examples of how to solve the task:
--- Start Example 1 ---
User query: "Show me all construction sites in Wuppertal and Düsseldorf that have May as their start month."
Answer:
{{
  "search_terms": [
    "construction site Wuppertal",
    "construction site Düsseldorf",
    "construction site start month"
  ]
}}
--- End Example 1 ---

--- Start Example 2 ---
User query: "Give me all suspended railway stops in Wuppertal that contain 'Straße' or 'Strasse' in their name"
Answer:
{{
  "search_terms": [
    "suspended railway stop Wuppertal",
    "suspended railway stop name"
  ]
}}
--- End Example 2 ---

--- Start Example 3 ---
User query: "Return all boreholes and their company names, that are deeper than 500 meters and located in Norway."
Answer:
{{
  "search_terms": [
    "borehole Norway",
    "borehole company name",S
  ]
}}
--- End Example 3 ---

User query: "{query}"

{final_output_instructions}

{optional_few_shot_examples}
""")







# Master template for Step 4: Generate SPARQL query
SPARQL_GENERATOR_PROMPT_MULTI_MODEL = ChatPromptTemplate.from_template("""
This is the fourth sub-step in the process of converting a user query into a SPARQL query.

The basis is a multitude of individual datasets. For each dataset, there is a separate semantic model that describes the underlying data structure.
Your task is to use the collected information to generate a SPARQL query that answers the user query.

Step 1: You receive the user query and a list of semantic models that have been identified as relevant for the query.
Analyze the semantic models in relation to the query and first identify the sought results of the query and which variables must therefore be the output.
Step 2: Then consider whether a calculation operation (e.g., COUNT, SUM, ORDER BY) is necessary to answer the query or whether it is only about the extraction of filtered instance data. Only use calculation operations if the calculation is really necessary and there are no data points that can be extracted directly. Make sure to always convert string values to numbers if necessary for calculations (e.g., SUM). Boolean expressions are also represented as strings with "1" for True and "0" for False.
Step 3: Identify classes whose instances must be stored in an variable via a class assignment (e.g., ?variable a local:Klasse) to avoid confusion with classes that are connected to a parent class through the same property. 
Step 4: Connect the target variables with the auxiliary variables by meaningfully connecting them with the properties from the given semantic models. CAUTION: DO NOT USE CLASSES AS AN OBJECT OR SUBJECT OF A TRIPLE, ONLY VARIABLES. Only use classes for type declarations of variables (i.e., ?variable a prefix:class). 
Step 5: If you also need information from the RDF data that is not mapped via the mapping or the semantic model, use a FILTER clause to extract this information.

Rules:
- Caution: Only create ONE single query that answers the user query by combining information from the provided triple blocks.
- If necessary, use `UNION` to combine results from different semantic models. But only do this if it is not possible in one query because the models differ too much.
- Only use correct designations from the respective triple blocks of the associated models, both for classes, predicates, and objects.
- Do not shorten the query to keep it clear and error-free.
- Use the correct prefixes and make sure to always name classes correctly and filter by classes if necessary, e.g., always ?XY a local:Fahrradabstellanlage, as otherwise there may be too many entries.
- All actual values are stored as the type depicted in the semantic model (e.g. xsd:string -> string) and may need to be explicitly converted to numbers, for example when SUM is used. Boolean expressions are also represented as strings with "1" for True and "0" for False.
- IMPORTANT: If a predicate can have multiple target objects, ALWAYS filter by type as specified in the semantic model.
- IMPORTANT: Only use classes as a type declaration of a variable (i.e. ?variable a prefix:class). In all other cases Subject and Object of the triple need to be variables (i.e. ?variable1 prefix:predicate ?variable2). 
{optional_instance_data_instructions}

- The following prefixes are predefined and are considered present. They serve only as assistance. You do not need to specify them in the query:
PREFIX owl:    <http://www.w3.org/2002/07/owl#>
PREFIX plasma: <http://plasma.uni-wuppertal.de/ontology#>
PREFIX plcm:   <http://plasma.uni-wuppertal.de/cm#>
PREFIX plsm:   <http://plasma.uni-wuppertal.de/sm/>
PREFIX rdf:    <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs:   <http://www.w3.org/2000/01/rdf-schema#>
PREFIX xsd:    <http://www.w3.org/2001/XMLSchema#>
PREFIX eno:   <http://example.org/ontology/energy#>
PREFIX enoptl:    <http://example.org/ontology/energy-ptl#>

{optional_few_shot_examples}

Now create ONE SINGLE valid SPARQL query based on the following information to answer the user query.
User query:
"{query}"
For the query, the following semantic model types were identified as relevant and the corresponding models were selected. Each block contains the triples for a specific model:
{model_info_blocks}
Notes from the model check (information about what each model can cover or not):
{model_check_hints}

{instance_data_info}

{final_output_instructions}

""")

# Output instruction for Step 4
OUTPUT_INSTRUCTIONS_GENERATE_SPARQL = """
Now provide ONLY the complete SPARQL query starting from SELECT - no explanation and no surrounding quotation marks or Markdown code blocks."""




# === NEW PROMPTS FOR SPLIT VALIDATION ===

# STEP 3a: Model selection with instance filter check
OUTPUT_INSTRUCTIONS_SELECT_MODELS_WITH_INSTANCE_FILTER = (
    "Provide your answer exclusively in the following JSON format - without greeting, explanation, or any other text:\n"
    "{\n  \"selected_models\": [\n    \"filename_of_first_selected_model.ttl\",\n    \"filename_of_second_selected_model.ttl\"\n  ],\n"
    "  \"evaluation\": \"fully_possible\",\n"
    "  \"reasoning\": \"A detailed justification for your evaluation.\",\n"
    "  \"searched_terms\": [ \"only fill in for instance_filter_required\" ],\n"
    "  \"instance_filter_hints\": \"only fill in for instance_filter_required\"\n}"
)

SELECT_MODELS_WITH_INSTANCE_FILTER_PROMPT_TEMPLATE = ChatPromptTemplate.from_template("""
You are an expert in selecting and analyzing semantic data models. Your task is to select the suitable models from a list of candidate models for a user query and evaluate whether this query can be fully answered.

YOUR TASKS:

1. SELECTION: Analyze the user query and the CONTENT of the models provided below. Select from the list the filenames of the models that are absolutely necessary to answer the query.
    Proceed as follows:
    Step 1: Identify the sought results of the query and which variables must therefore be the output.
    Step 2: Check which conditions are imposed on these variables that could limit the result (e.g., only at this time, only at this location, etc.)
    Step 3: Analyze the provided semantic models and their triples with regard to whether they contain the classes, predicates, or literals that are needed to represent the sought results and conditions.
    Step 4: Decide which models are absolutely needed to cover all aspects of the query. If it is possible to answer the query with fewer semantic models, then select only those. That means be minimalistic in the selection.
2. VALIDATION: Subsequently evaluate whether a SPARQL query can be generated with the semantic models you selected that answers the user query.

IMPORTANT - ANSWER TWO SEPARATE QUESTIONS:

QUESTION A: Are the model structures (classes/properties) complete?
QUESTION B: Do you need information on specific values that need to searched for in the data instances as they are not part of the semantic model?

Additional rules:
- Make sure to only select models that are absolutely necessary to answer the query.
- If multiple models cover the same aspects, but each has relevant differences, select all of them.
- If the user query is looking for a specific type of entity, only select models that contain this entity type. If the user query is looking for a general entity type only select the model with the superordinate class of the entity and not the various subclasses. 
- Central data requirements in the user query must be present in the selected models as a subjekt and not just as an object.

You have the following evaluation options:

   - fully_possible:
     * The model structures (classes/properties) are complete AND
     * NO specific values need to be searched in the data
     * OR: All required values are already present as literals in the model triples

   - instance_filter_required:
     * The model structures are complete BUT
     * The query contains specific filter values (specific IDs like "86035", specific years like "2015", specifc names like "Sebastian Müller") that are NOT visible in the model triples
     * These values must be searched in the RDF data instances
     * IMPORTANT: These values can not be general categories that are mentioned in the query, but specifc values connected to a class/property in the semantic model
     * IMPORTANT: Even if the models are "complete", instance search may still be necessary!

   - not_possible:
     * Basic concepts (classes/properties) are completely missing in the models

EXAMPLES for fully_possible (NO value search necessary):
- "Show all materials" → No specific values searched
- "CO2 emission of materials" → No specific values searched
- "Materials in Wuppertal" where model contains: `local:location "Wuppertal"^^xsd:string` → Value already visible in triples

EXAMPLES for instance_filter_required (value search NECESSARY):
example semantic_model_40.ttl:
@prefix local: <https://local.ontology#> .
@prefix plasma: <http://plasma.uni-wuppertal.de/ontology#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

local:Stra_enbahnstrecke local:angefahren_von local:Verkehrslinie ;
    local:befindet_sich_an local:Geographische_Linie ;
    local:identifiziert_durch local:Universally_Unique_Identifier__UUID_ ;
    local:ist_Teil_von local:Schienennetz ;
    local:located_in "Rostock"^^xsd:string .

local:Geographische_Breite plasma:hasValue xsd:string .

local:Geographische_L_nge plasma:hasValue xsd:string .

local:Geographische_Linie local:besteht_aus local:Geographischer_Punkt .

local:Geographischer_Punkt local:besteht_aus local:Geographische_Breite,
        local:Geographische_L_nge .

local:Universally_Unique_Identifier__UUID_ plasma:hasValue xsd:string .

local:Verkehrslinie plasma:hasValue xsd:string ;
    local:ist_vom_Typ local:Stra_enbahn .

User query example: Tell me which traffic lines operate on the tram route with ID e2ce9b90-b5ee-11ea-94ce-005056b3c6e4 in Rostock.
Output:
 "{{\n  \"selected_models\": [\n    \"semantic_model_40.ttl\"  ],\n"
    "  \"evaluation\": \"instance_filter_required\",\n"
    "  \"reasoning\": \"Model semantic_model_40.ttl contains the required classes and predicates, but the specific ID must be searched for in the data instances.\",\n"
    "  \"searched_terms\": [ \"e2ce9b90-b5ee-11ea-94ce-005056b3c6e4\" ],\n"
    "  \"instance_filter_hints\": \"The ID e2ce9b90-b5ee-11ea-94ce-005056b3c6e4 is not found in the semantic model and it must be checked whether the data instance exists\"\n}}"

DECISION LOGIC:
1. Check: Are all classes/properties for the query present?
   → NO: evaluation = "not_possible"
   → YES: Continue to step 2

2. Check: Does the query contain specific filter values (specific IDs like "86035", specific years like "2015", specifc names like "Sebastian Müller")?
   → NO: evaluation = "fully_possible", searched_terms = []
   → YES: Continue to step 3

3. Check: Are these filter values already visible as literals in the model triples?
   → YES (all values visible): evaluation = "fully_possible", searched_terms = []
   → NO (values missing): evaluation = "instance_filter_required", searched_terms = [list of missing values]

FOR instance_filter_required:
- Fill `searched_terms` with the specific values from the query (specific IDs like "86035", specific years like "2015", specifc names like "Sebastian Müller") that are NOT in the model triples
- Fill `instance_filter_hints` with explanation of which values must be searched and why
- NEVER leave searched_terms empty when evaluation = "instance_filter_required"!

================================================================================
EXAMPLES:
================================================================================

Here is an example of how to solve the task:

--- Start Example 1 ---
User query: "Overview of traffic light systems in Hamburg, including mounting location and direction."
Triples of semantic models identified as relevant:

Model: 003.ttl
```
local:Fahrtrichtungspfeil local:ausrichtung local:Gradma_ .
local:Fahrtrichtungspfeil local:befindet_sich_an local:Geographisches_Polygon .
local:Fahrtrichtungspfeil local:hat local:Fahrtrichtungspfeiltyp .
local:Fahrtrichtungspfeil local:located_in "Rostock"^^xsd:string .
local:Geographischer_Punkt local:besteht_aus local:UTM_false_easting .
local:Geographischer_Punkt local:besteht_aus local:UTM_northing .
local:Geographisches_Polygon local:besteht_aus local:Geographischer_Punkt .
local:Geographisches_Polygon local:koordinatenreferenzsystem xsd:string .
local:Gradma_ local:absolute_Bezugsrichtung local:Geographischer_Norden .
```

Model: 020.ttl
```
local:CarSharing_Station local:befindet_sich_an local:Geographischer_Punkt .
local:CarSharing_Station local:betrieben_durch local:Cambio .
local:CarSharing_Station local:bietet_an local:Fahrzeug .
local:CarSharing_Station local:hat local:Adresse .
local:CarSharing_Station local:hat local:Bezeichnung .
local:CarSharing_Station local:hat local:Standortbezeichnung .
local:CarSharing_Station local:hat local:Zugangsinformation .
local:CarSharing_Station local:hat local:Zusatzinformation .
local:CarSharing_Station local:identifiziert_durch local:Identifikator .
local:CarSharing_Station local:located_in "Berlin"^^xsd:string .
local:Adresse local:besteht_aus local:Hausnummer .
local:Adresse local:besteht_aus local:Land .
local:Adresse local:besteht_aus local:Postleitzahl .
local:Adresse local:besteht_aus local:Stadt .
local:Adresse local:besteht_aus local:Stra_e .
local:Fahrzeug local:geh_rt_zu local:Kategorie .
local:Geographischer_Punkt local:besteht_aus local:Geographische_Breite .
local:Geographischer_Punkt local:besteht_aus local:Geographische_L_nge .
local:Kategorie local:identifiziert_durch local:Identifikator .
```

Model: 078.ttl
```
local:Lichtsignalanlage local:befindet_sich_an local:Geographischer_Punkt .
local:Lichtsignalanlage local:erfasst_am local:Zeitstempel .
local:Lichtsignalanlage local:hat local:Befestigungsort .
local:Lichtsignalanlage local:hat local:H_he .
local:Lichtsignalanlage local:hat local:Richtung .
local:Lichtsignalanlage local:hat local:Verwendungszweck .
local:Lichtsignalanlage local:identifiziert_durch local:Identifikator .
local:Lichtsignalanlage local:identifiziert_durch local:Lichtsignalanlagen_Nummer .
local:Lichtsignalanlage local:located_in "Hamburg"^^xsd:string .
local:Befestigungsort local:identifiziert_durch local:Identifikator .
local:Ellipsoidische_H_he local:hat local:Standardabweichung .
local:Geographischer_Punkt local:besteht_aus local:Ellipsoidische_H_he .
local:Geographischer_Punkt local:besteht_aus local:UTM_false_easting .
local:Geographischer_Punkt local:besteht_aus local:UTM_northing .
local:Geographischer_Punkt local:koordinatenreferenzsystem xsd:string .
local:Richtung local:ausrichtung local:Geographischer_Norden .
local:Richtung local:gemessen_in local:Gradma_ .
local:UTM_false_easting local:hat local:Standardabweichung .
local:UTM_northing local:hat local:Standardabweichung .
local:Zeitstempel local:zeitstempelformat xsd:string .
```

Answer:
{{
  "selected_models": ["078.ttl"],
  "evaluation": "fully_possible",
  "reasoning": "Model 078.ttl contains the class 'local:Lichtsignalanlage' with the required properties 'local:hat local:Befestigungsort' and 'local:hat local:Richtung' as well as the geographic assignment to Hamburg via 'local:located_in Hamburg'. All aspects of the user query (overview of traffic light systems in Hamburg including mounting location and direction) can be fully covered. The other models (003.ttl for directional arrows in Rostock and 020.ttl for car sharing stations in Berlin) are not relevant for this query."
}}

--- End Example 1 ---

--- Start Example 2 ---

User query: "Give me all suspended railway stops in Wuppertal including construction year"
Triples of semantic models identified as relevant:

Model: 061.ttl
```
local:Stra_enbeleuchtung local:befindet_sich_an local:Geographischer_Punkt .
local:Stra_enbeleuchtung local:identifiziert_durch local:Identifikator .
local:Stra_enbeleuchtung local:identifiziert_durch local:Standortbezeichnung .
local:Stra_enbeleuchtung local:located_in "Duisburg"^^xsd:string .
local:Geographischer_Punkt local:besteht_aus local:UTM_false_easting .
local:Geographischer_Punkt local:besteht_aus local:UTM_northing .
local:Geographischer_Punkt local:koordinatenreferenzsystem xsd:string .
```




Model: 099.ttl
```
local:Schwebebahnstation local:angefahren_von local:Wuppertaler_Schwebebahn .
local:Schwebebahnstation local:befindet_sich_an local:Geographischer_Punkt .
local:Schwebebahnstation local:hat local:Bezeichnung .
local:Schwebebahnstation local:ist_ein_e_ local:Haltestelle .
local:Schwebebahnstation local:located_in "Wuppertal"^^xsd:string .
local:Geographischer_Punkt local:besteht_aus local:Geographische_Breite .
local:Geographischer_Punkt local:besteht_aus local:Geographische_L_nge .
local:Geographischer_Punkt local:koordinatenreferenzsystem xsd:string .
```



Model: 064.ttl
```
local:Haltestelle local:benutzt_von local:Zug .
local:Haltestelle local:hat local:Bezeichnung .
local:Haltestelle local:hat local:Z_hlung .
local:Haltestelle local:identifiziert_durch local:Haltestellennummer .
local:Haltestelle local:located_in "Berlin"^^xsd:string .
local:Aussteiger local:ist_ein_e_ local:Mensch .
local:Einsteiger local:ist_ein_e_ local:Mensch .
local:Tag local:befindet_sich_in local:Jahr .
local:Tag local:durchschnitt local:Aussteiger .
local:Tag local:durchschnitt local:Einsteiger .
local:Tag local:ist_vom_Typ local:Tagestyp .
local:Z_hlung local:bezieht_sich_auf local:Tag .
```

Answer:
{{
  "selected_models": ["099.ttl"],
  "evaluation": "partially_possible",
  "reasoning": "Model 099.ttl contains the class 'local:Schwebebahnstation' with the required property 'local:hat local:Bezeichnung' as well as the geographic assignment to Wuppertal via 'local:located_in Wuppertal'. The suspended railway stops can be identified and queried. However, the information about the construction year of the stops is missing in the semantic model, which means this aspect of the query cannot be covered.",
  "user_clarification": "The semantic model does not contain information about the construction year of the suspended railway stops. Therefore, the construction year cannot be considered in the query. Should the SPARQL query still be created to query all suspended railway stops in Wuppertal (without the construction year information)?"
}}

--- End Example 2 ---

================================================================================
NOW ANALYZE THIS SPECIFIC REQUEST:
================================================================================

USER QUERY:
"{query}"

CANDIDATE MODELS:
{candidate_models_content}

OUTPUT FORMAT:
{final_output_instructions}

{optional_few_shot_examples}
""")

# STEP 3b: Final model validation
OUTPUT_INSTRUCTIONS_FINAL_MODEL_VALIDATION = (
    "Provide your answer exclusively in the following JSON format - without greeting, explanation, or any other text:\n"
    "{\n  \"final_evaluation\": \"fully_possible or not_possible\",\n"
    "  \"final_reasoning\": \"A detailed justification for your final evaluation. If multiple files were validated, explain explicitly why these are needed to answer the user query and why it cannot be done with fewer.\",\n"
    "  \"corrected_models\": [\"list_of_final_model_filenames\"]\n}"
)

FINAL_MODEL_VALIDATION_PROMPT_TEMPLATE = ChatPromptTemplate.from_template("""
You are an expert in semantic data models and SPARQL query generation. This is the final validation step after model selection and optional instance search.

YOUR TASK:

Make a final decision whether a meaningful SPARQL query can be generated with the available information (semantic models and instances) that answers the user query.
Your task here is in particular to perform a redundancy check. That means, go through the selected models and check whether the query can also be answered with fewer models.
Consider which file the provided instance data comes from.

If the query can be answered with fewer models, return the reduced list of actually needed model filenames in "corrected_models".
If all selected models are needed, return all originally selected model filenames.

Additional rules:
- Make sure to only select models that are absolutely necessary to answer the query.
- If multiple models cover the same aspects, but each has relevant differences, select all of them.
- If the user query is looking for a specific type of entity, only select models that contain this entity type. If the user query is looking for a general entity type only select the model with the superordinate class of the entity and not the various subclasses. 
- Central data requirements in the user query must be present in the selected models as a subjekt and not just as an object.

EVALUATION OPTIONS:
- fully_possible: A SPARQL query can be generated that answers the query
- not_possible: Basic concepts are completely missing or the query cannot be implemented with SPARQL

CONSIDER IMPORTANT SPARQL CAPABILITIES:

1. Aggregation: COUNT, SUM, AVG can calculate counts/sums, even without explicit count properties
2. String matching: FILTER with CONTAINS can search in literal values, even without perfect model mapping
3. Type filtering: `?x a local:Klasse` can filter entities by type
4. Datatype conversion: xsd:integer can convert strings to numbers
5. Text search: Addresses, names and IDs can be found via literal matching
6. Relationship navigation: Properties can be chained to go deep into the data structure

================================================================================
EXAMPLES:
================================================================================

Here are examples of how to perform the final validation:

--- Start Example 1 ---
User query: "Show me all traffic light systems in Hamburg including mounting location and direction."

Selected models:
Model: 078.ttl
```
local:Lichtsignalanlage local:befindet_sich_an local:Geographischer_Punkt .
local:Lichtsignalanlage local:hat local:Befestigungsort .
local:Lichtsignalanlage local:hat local:Richtung .
local:Lichtsignalanlage local:located_in "Hamburg"^^xsd:string .
```

Instance search results: No instance search was performed (evaluation was "fully_possible").

Answer:
{{
  "final_evaluation": "fully_possible",
  "final_reasoning": "The selected model 078.ttl contains all required classes and properties to answer the query. The class 'local:Lichtsignalanlage' has the properties 'local:hat local:Befestigungsort' and 'local:hat local:Richtung', and the location filter 'Hamburg' is present in the model. A SPARQL query can be generated to retrieve all traffic light systems with their mounting locations and directions in Hamburg. No additional models are needed.",
  "corrected_models": ["078.ttl"]
}}
--- End Example 1 ---

--- Start Example 2 ---
User query: "Give me the traffic lines operating on the tram route with ID e2ce9b90-b5ee-11ea-94ce-005056b3c6e4 in Rostock."

Selected models:
Model: 040.ttl
```
local:Stra_enbahnstrecke local:angefahren_von local:Verkehrslinie .
local:Stra_enbahnstrecke local:identifiziert_durch local:Universally_Unique_Identifier__UUID_ .
local:Stra_enbahnstrecke local:located_in "Rostock"^^xsd:string .
local:Universally_Unique_Identifier__UUID_ plasma:hasValue xsd:string .
local:Verkehrslinie plasma:hasValue xsd:string .
```

Instance search results: Instance search was performed. Found 1 matching instance for UUID "e2ce9b90-b5ee-11ea-94ce-005056b3c6e4" in model 040.ttl. The instance exists in the data.

Answer:
{{
  "final_evaluation": "fully_possible",
  "final_reasoning": "The model 040.ttl contains the class 'local:Stra_enbahnstrecke' with the property 'local:angefahren_von local:Verkehrslinie' which allows us to retrieve the traffic lines operating on a specific tram route. The UUID can be used to filter for the specific route. Instance search confirmed that the requested UUID exists in the data. A SPARQL query can be generated using FILTER on the UUID value and retrieving the associated traffic lines. Only this single model is needed.",
  "corrected_models": ["040.ttl"]
}}
--- End Example 2 ---

--- Start Example 3 ---
User query: "Count all bicycle parking facilities in Berlin with more than 10 parking spaces."

Selected models:
Model: 015.ttl
```
local:Fahrradabstellanlage local:anzahl local:Stellplatz .
local:Fahrradabstellanlage local:befindet_sich_an local:Geographischer_Punkt .
local:Fahrradabstellanlage local:located_in "Berlin"^^xsd:string .
local:Stellplatz plasma:hasValue xsd:string .
```

Model: 022.ttl
```
local:Parkhaus local:anzahl local:Stellplatz .
local:Parkhaus local:located_in "Berlin"^^xsd:string .
```

Instance search results: No instance search was performed.

Answer:
{{
  "final_evaluation": "fully_possible",
  "final_reasoning": "After redundancy check, only model 015.ttl is needed. It contains 'local:Fahrradabstellanlage' with property 'local:anzahl local:Stellplatz' which allows counting and filtering by number of parking spaces. The location 'Berlin' is present. A SPARQL query can use COUNT aggregation and FILTER with xsd:integer conversion to filter for more than 10 parking spaces. Model 022.ttl is about parking garages (Parkhaus), not bicycle parking facilities, so it is not relevant for this query and should be removed from the final list.",
  "corrected_models": ["015.ttl"]
}}
--- End Example 3 ---

================================================================================
NOW PERFORM FINAL VALIDATION FOR THIS SPECIFIC REQUEST:
================================================================================

USER QUERY:
"{query}"

SELECTED MODELS:
{selected_models_content}

INSTANCE SEARCH RESULTS:
{instance_search_summary}

OUTPUT FORMAT:
{final_output_instructions}

IMPORTANT NOTES:
- Think practically: What can SPARQL accomplish, not what the model maps perfectly
- Consider that data can be accessed via string matching and aggregation
- Focus on the evaluation: Can a meaningful SPARQL query be generated?

{optional_few_shot_examples}
""")


LLM_SPARQL_QUERY_EDIT_EVALUATION_PROMPT_STEP_1 = ChatPromptTemplate.from_template("""
You are a SPARQL repair system within an evaluation process for a system that converts natural language queries into SPARQL queries.

Your task is to analyze, correct, and describe corrections for a SPARQL query generated by an LLM so that it fulfills the same function as a correctly functioning ground truth query, i.e., delivers the same result.
Variable names may differ as long as the underlying functionality is the same.

Your goal is NOT to completely align the LLM query with the content, but to use the ground truth query as support to correct the generated one so that it delivers the desired result.
So you should understand from the ground truth query, the triples from the semantic models and the hints, how exactly the user query should be answered.

The following hints or rules existed for the original SPARQL generation:
Rules:
- The goal was to create ONE query that answers the user query by combining information from the provided triple blocks.
- If necessary, use `UNION` to combine results from different parts of the WHERE clause.
- Use SELECT DISTINCT to avoid duplicates and make sure to choose the correct variables with SELECT.
- Only use correct designations from the respective triple blocks of the associated models, both for classes, predicates, and objects.
- Do not shorten the query to keep it clear and error-free.
- Use the correct prefixes and make sure to always name classes correctly and filter by classes if necessary, e.g., always ?XY a local:Haus, as otherwise there may be too many entries.
- Observe the structure of the data that the semantic model specifies and go deep enough in the query to retrieve the sought value.
- Pay attention to what is REALLY being searched for and use the correct classes and predicates from the triple blocks.
- IMPORTANT: If a predicate can have multiple target objects, ALWAYS filter by type as specified in the semantic model. When filtering by cities, use the triple local:located_in "Stadt"^^xsd:string .
- IMPORTANT: To obtain the actual value (e.g., a number, text, date) from an entity that represents this value, you must ALWAYS use the predicate plasma:hasValue. Otherwise you will only get the object in the result, not the sought value. You must actually use plasma:hasValue for everything that is being asked for.
- All actual values are stored as strings and may need to be explicitly converted to numbers, for example when SUM is used.

Procedure now:
1. Read the ground truth query carefully.
2. Explain in your own words what functionality it fulfills. What does the query do? What data does it filter or extract?
3. Compare the functionality of the LLM query with it: What information is missing or incorrectly implemented?
4. Find the errors in the generated query and consider how to correct them.
5. Then determine the minimally necessary and clearly distinguishable change steps so that the LLM query fulfills the same functionality as the ground truth query.
   Focus on individual, logical corrections.
6. Note the changes in the form of a list of clear, precise steps.
7. Really list ALL changes that change the functionality or influence the generated result in terms of content. But name ALL changes that correspond to this.
8. Create a corrected SPARQL query based on the collected changes.

Examples of valid steps:
- "Replace variable ?Name with ?NameWert in SELECT structure"
- "Add missing triple `?Adresse a local:Adresse`"
- "Replace incorrect predicate `local:stadtname` with `local:located_in`"
- "Add missing FILTER `FILTER(xsd:integer(?wert) > 4)`"
- "Add COUNT structure `COUNT(?station)` with `AS ?anzahl`"
- "Add BIND expression `BIND(xsd:integer(?str) AS ?int)`"
- "Add UNION structure between two blocks"

Important:
- Only return steps that are functionally necessary, ignore cosmetic differences such as variable names.
- You may add, change or remove triples, filters or operators as long as the result is correct.
- Formulate the steps so that they can be used as a basis for a subsequent evaluation. Follow the examples.
- The ground truth query serves only as assistance and as orientation for functionality, you do not need to adapt variable names to ground truth if it is not required.
- When considering the result, data types do not matter, so binding is not mandatory.
- Also do not take over comments from the ground truth query

IMPORTANT OUTPUT FORMAT:
Your entire answer must be a single, valid JSON object. Your answer must absolutely not contain any text, comments or Markdown code blocks like ```json outside the JSON object. Start directly with `{{` and end with `}}`.

Structure of the JSON object:
{{
  "groundtruth_analysis": "[Your paragraph explaining the function of the ground truth query]",
  "llm_query_deviation": "[Your paragraph describing the deviations of the LLM query]",
  "changes": [
    "Description of functionally necessary change 1",
    "Description of change 2"
  ],
  "corrected_query": "SELECT ... WHERE {{{{ ... }}}}"
}}

EXAMPLE FOR INPUT AND OUTPUT:
User query: "I'm looking for all suspended railway stops that contain 'Straße' or 'Strasse'. Just list the names."
Semantic model triples: local:Schwebebahnstation a local:Haltestelle ; local:hat ?Bezeichnung ; local:located_in "Wuppertal"^^xsd:string .
Ground truth query for comparison: SELECT DISTINCT ?Name WHERE {{{{ ?station a local:Schwebebahnstation ; local:hat ?Bezeichnung . ?Bezeichnung plasma:hasValue ?RoherWert . BIND(LCASE(?RoherWert) AS ?NameLower) FILTER (CONTAINS(?NameLower, \\"straße\\") || CONTAINS(?NameLower, \\"strasse\\")) BIND(?RoherWert AS ?Name) }}}}
LLM query: SELECT ?Bezeichnung WHERE {{{{ ?station a local:Schwebebahnstation . ?station local:hat ?Bezeichnung . FILTER (CONTAINS(?Bezeichnung, \\"Straße\\") || CONTAINS(?Bezeichnung, \\"Strasse\\")) }}}}


EXAMPLE ANSWER (your output must exactly follow this JSON format):
{{
  "groundtruth_analysis": "The query finds all suspended railway stops whose designation string (retrieved via plasma:hasValue) contains 'straße' or 'strasse' in any case. It outputs the original string values of the designations.",
  "llm_query_deviation": "The LLM query filters directly on the entity ?Bezeichnung (instead of its string value) and performs a case-sensitive search. Additionally, the entity itself (not the string value) is output.",
  "changes": [
    "Add missing triple `?Bezeichnung plasma:hasValue ?RoherWert`",
    "Replace FILTER on entity with FILTER on string value",
    "Implement case-insensitive search using LCASE and lowercase comparison",
    "Output the original string value (?RoherWert as ?Name) instead of the entity"
  ],
  "corrected_query": "SELECT DISTINCT ?Name\\nWHERE {{{{ \\n  ?station a local:Schwebebahnstation ;\\n        local:hat ?Bezeichnung .\\n  ?Bezeichnung plasma:hasValue ?RoherWert .\\n  BIND(LCASE(?RoherWert) AS ?NameLower)\\n  FILTER (CONTAINS(?NameLower, \\"straße\\") || CONTAINS(?NameLower, \\"strasse\\"))\\n  BIND(?RoherWert AS ?Name)\\n}}}}"
}}

Proceed as instructed.
Here are the original hints for SPARQL query generation again:
Rules:
- The goal was to create ONE query that answers the user query by combining information from the provided triple blocks.
- If necessary, use `UNION` to combine results from different parts of the WHERE clause.
- Use SELECT DISTINCT to avoid duplicates and make sure to choose the correct variables with SELECT.
- Only use correct designations from the respective triple blocks of the associated models, both for classes, predicates, and objects.
- Do not shorten the query to keep it clear and error-free.
- Use the correct prefixes and make sure to always name classes correctly and filter by classes if necessary, e.g., always ?XY a local:Haus, as otherwise there may be too many entries.
- Pay attention to what is REALLY being searched for and use the correct classes and predicates from the triple blocks.
- IMPORTANT: If a predicate can have multiple target objects, ALWAYS filter by type as specified in the semantic model. When filtering by cities, use the triple local:located_in "Stadt"^^xsd:string .
- All actual values are stored as strings and may need to be explicitly converted to numbers, for example when SUM is used.
- IMPORTANT: To obtain the actual value (e.g., a number, text, date) from an entity that represents this value, you must ALWAYS use the predicate plasma:hasValue. Otherwise you will only get the object in the result, not the sought value.

Here are the collected and generated elements
It remains important to list all changes.
Procedure now:
1. Read the given elements carefully.
2. Explain in your own words what functionality it fulfills. What does the query do? What data does it filter or extract?
3. Compare the functionality of the LLM query with it: What information is missing or incorrectly implemented?
4. Then determine the minimally necessary and clearly distinguishable change steps so that the LLM query generates the same result as the ground truth query. Output variable names are not relevant.
5. Note the changes in the form of a list of clear, precise steps. The justifications for changes should only be based on the query and not on the ground truth query.
6. Really only list the changes that change the functionality or influence the generated result in terms of content. But name ALL changes that correspond to this.
7. Create a corrected SPARQL query based on the collected changes.
IMPORTANT: If possible use the same variable names as in the original LLM query to keep it comparable.

User query:
{query}

Semantic model triples:
{model_info_blocks}

Model hints:
{model_check_hints}

Ground truth query for comparison:
{groundtruth_sparql_query}

LLM query:
{generated_sparql_query}

Result snippets (to illustrate the difference):

Expected result (from ground truth query, with LIMIT 5):
Generated json
{groundtruth_query_result_snippet}

Actual result (from incorrect LLM query, with LIMIT 5):
Generated json
{generated_query_result_snippet}

""")



CONTINUE_SPARQL_GENERATION_PROMPT = ChatPromptTemplate.from_template("""
You are an assistant in a system that converts user queries into SPARQL. A previous stage has analyzed a user query and determined that it can only be partially answered with the available data. A follow-up question was asked to the user.

Here is the entire context:

1.  Original user query: "{original_user_query}"
2.  Analysis of the previous stage: "{initial_llm_reasoning}"
3.  Follow-up question to the user: "{clarification_question}"
4.  User's response: "{user_response}"

Your task is to decide, based on the user's response, whether SPARQL query generation should continue.

Rules:
- If the user's response is positive or affirmative (e.g., "Yes", "continue", "ok", "execute anyway"), respond with "YES".
- If the user's response is negative or declining (e.g., "No", "leave it", "cancel"), respond with "NO".
- Try to understand the user's intent, even if the response is not exactly "Yes" or "No".

Respond ONLY with the word "YES" or "NO". Do not provide any other text.
""")


# --- ADAPTIVE FEW-SHOT PROMPTING TEMPLATES ---

OPERATOR_PREDICTION_PROMPT_TEMPLATE = ChatPromptTemplate.from_template("""
You are an expert in SPARQL query generation. Your task is to predict the likely needed SPARQL operators based on a natural language user query and the available semantic models.

USER QUERY:
"{query}"

AVAILABLE SEMANTIC MODELS:
{model_info_blocks}

AVAILABLE SPARQL OPERATORS:
{available_operators}

IMPORTANT NOTES:
- Analyze the user query for various aspects:
  * Are specific values/texts searched for? → SELECT
  * Should filtering be applied? → FILTER, CONTAINS
  * Are aggregations needed? → COUNT, SUM, AVG
  * Should duplicates be avoided? → DISTINCT
  * Are multiple entity types combined? → UNION
  * Are optional data involved? → OPTIONAL
  * Is sorting or grouping needed? → ORDER BY, GROUP BY
  * Are strings compared? → LCASE, CONTAINS
  * Are data types converted? → xsd:integer
  * Are new variables created? → BIND

- Consider the structure of the semantic models
- Note that type filtering is often necessary (e.g., "?x a local:KlasseName")

EXAMPLES OF TYPICAL PATTERNS:
- "All X that contain Y" → SELECT, DISTINCT, FILTER, CONTAINS, LCASE
- "X with more than N" → SELECT, FILTER, xsd:integer
- "Sorted list of X" → SELECT, ORDER BY
- "X and Y together" → SELECT, UNION, BIND

{final_output_instructions}

""")

OUTPUT_INSTRUCTIONS_PREDICT_OPERATORS = (
    "Respond exclusively with a JSON object in the following format:\n"
    "{\n  \"predicted_operators\": [\"OPERATOR1\", \"OPERATOR2\", \"OPERATOR3\"],\n"
    "  \"reasoning\": \"Brief justification for the prediction\"\n}\n\n"
    "Select 3-8 of the most likely operators."
)


# === EMPTY RESULTS CORRECTION PROMPT ===

EMPTY_RESULTS_CORRECTION_PROMPT = ChatPromptTemplate.from_template("""
You are a SPARQL expert. The generated SPARQL query was executed successfully, but returned NO results (empty result set).

Your task is to analyze whether the empty result is LEGITIMATE or if the query should be corrected.

================================================================================
DECISION CRITERIA:
================================================================================

Empty results are LEGITIMATE for:
- Existence questions ("Are there any...?", "Does... exist?") where the answer is "no"
- Questions with conditions that genuinely don't match any data ("All people over 200 years old" when none exist)
- Queries explicitly asking for non-existent data or testing for absence

Empty results should be CORRECTED for:
- Wrong properties or classes used (check against semantic models!)
- Faulty triple connections (e.g., subject-object swapped)
- Too restrictive or incorrect FILTER conditions
- Missing OPTIONAL clauses where needed
- Incorrect URI construction or property paths
- Missing type filters (e.g., `?x a local:ClassName`)
- The query structure expects results based on the user's intent

================================================================================
SPARQL GENERATION GUIDELINES (if correction is needed):
================================================================================

Follow these steps to generate a correct SPARQL query:

Step 1: Identify the sought results and which variables must be the output
Step 2: Consider whether calculation operations (COUNT, SUM, ORDER BY) are necessary
Step 3: Identify classes that need explicit type assignment (e.g., `?x a local:ClassName`)
Step 4: Connect variables using properties from the semantic models
Step 5: Add FILTER clauses for additional constraints not in the model

IMPORTANT RULES:
- Use ONLY correct designations from the semantic model triple blocks
- ALWAYS filter by type if a predicate can have multiple target objects
- All literal values are stored as strings and may need conversion (e.g., xsd:integer for SUM)
- Boolean values are represented as strings: "1" for True, "0" for False
- Use UNION if combining results from different semantic models
- Always include `?x a local:ClassName` to filter by class type
- Check that property paths connect correctly (subject → property → object)
                                                                   
Information on instance data:
{instance_data_context}
Examples of correct SPARQL patterns:
{few_shot_examples_context}

PREDEFINED PREFIXES (do not include in your answer):
PREFIX owl:    <http://www.w3.org/2002/07/owl#>
PREFIX plasma: <http://plasma.uni-wuppertal.de/ontology#>
PREFIX plcm:   <http://plasma.uni-wuppertal.de/cm#>
PREFIX plsm:   <http://plasma.uni-wuppertal.de/sm/>
PREFIX rdf:    <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs:   <http://www.w3.org/2000/01/rdf-schema#>
PREFIX xsd:    <http://www.w3.org/2001/XMLSchema#>
PREFIX local:  <https://local.ontology#>
PREFIX eno:   <http://example.org/ontology/energy#>
PREFIX enoptl:    <http://example.org/ontology/energy-ptl#>

================================================================================
CURRENT PROBLEM - ANALYZE THIS SPECIFIC CASE:
================================================================================

USER QUERY:
"{user_query}"

SEMANTIC MODELS (Available classes, properties, and their relationships):
{model_info_blocks}

MODEL CHECK HINTS:
{model_check_hints}

CURRENT SPARQL QUERY (returned empty results):
{generated_sparql_query}

{previous_attempts_context}

================================================================================
ADDITIONAL DEBUGGING INFORMATION:
================================================================================

{decomposition_analysis_context}

**IMPORTANT DEBUGGING HINTS (PATTERN-ABLATION DIAGNOSTICS):**

STEP 1 - INCREMENTAL PATTERN ABLATION:
- Shows which pattern causes the query to return COUNT=0
- If a "killer pattern" is identified, that specific triple is problematic
- Fix by: correcting property name, checking property direction, or using correct class

STEP 2 - FILTER EFFECT ISOLATION:
- If a filter blocks all results, check the observed range
- Common issue: filter condition outside actual data range (e.g., "year >= 2008" when data only has 1990-2007)
- Fix by: adjusting filter values to match actual data range or removing impossible conditions

STEP 3 - JOIN KEY OVERLAP:
- If join variable has ZERO OVERLAP, you're joining on wrong property
- Common issue: joining on "name" instead of "id" (or vice versa)
- Fix by: using correct join property (check sample data for available properties)

STEP 4 - SAMPLE DATA:
- Shows actual property names and values from database
- Use this to verify exact property names (case-sensitive!)
- Check for whitespace, type mismatches, or URI format issues

GENERAL HINTS:
- Check the instance data to see exact string formats (whitespace matters!)
- Compare your query structure with the successful examples
- If string literals don't match, check spacing and case sensitivity

================================================================================
YOUR TASK:
================================================================================

1. Carefully analyze the semantic models and compare with the current query
2. Review the instance data to identify potential string/property mismatches
3. Compare with successful query patterns to identify structural issues
4. Decide if the empty result is legitimate or needs correction
5. If correction is needed, generate a corrected SPARQL query following the guidelines above

Answer in JSON format:
{{
  "should_correct": true/false,
  "justification": "Detailed explanation of why the empty result is legitimate OR which semantic errors were found (e.g., wrong property used, missing type filter, incorrect triple pattern)",
  "corrected_query": "Complete corrected SPARQL query starting from SELECT (only if should_correct=true, otherwise leave empty)"
}}

IMPORTANT:
- Provide ONLY the query body starting from SELECT - no prefixes, no explanations, no code blocks
- If should_correct=false, leave corrected_query as empty string
- Be thorough in your justification to explain your reasoning
""")

OUTPUT_INSTRUCTIONS_EMPTY_RESULTS_CORRECTION = """
Answer in JSON format:
{
  "should_correct": true/false,
  "justification": "Detailed explanation",
  "corrected_query": "Complete SPARQL query starting from SELECT"
}
"""


