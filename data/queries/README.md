# Question corpus and adaptive ground truth

56 question formulations derived from 22 intents (Section 5.2 of the paper):

| Condition | Questions | Construction |
|---|---|---|
| BASE | 17 | the question uses the vocabulary of the mappings verbatim; control condition |
| SYN | 17 | schema-related terms replaced by the wording a user without schema knowledge would choose (synonyms, hypernyms, circumlocutions, format changes, renamed entities); same ground truth as BASE |
| TYPO | 17 | entity or attribute references carry spelling errors; same ground truth as BASE |
| UNDER | 5 | the question omits a criterion or threshold only the user can supply; scored on whether the system reports the question as underspecified (Section 6.3) |

## Files

| File | Content |
|---|---|
| [experimental_corpus.py](experimental_corpus.py) | the corpus as Python objects: `SET_BASE`, `SET_SYN`, `SET_TYPO`, `SET_UNDER`, each entry an `AdaptiveGroundTruth` with the question, its admissible SPARQL readings, the column labels and notes. SYN and TYPO entries record their substitutions in `variant_description`. The module also defines five `CROSS` questions that are not part of the paper. |
| [corpus.json](corpus.json) | every formulation with all readings and column labels, exported from the module (`python scripts/export_corpus.py`) |
| [corpus.csv](corpus.csv) | one row per formulation: id, condition, dataset, BASE counterpart, number of readings, question, substitution |
| [use_case_queries_tiered_tuples.py](use_case_queries_tiered_tuples.py) | the `AdaptiveGroundTruth` and `ColumnDef` classes and the result cache of the readings |

## Ground-truth structure (Section 4.1)

Every question carries one or more admissible SPARQL readings (`sparql_queries`). A system answer is
scored against each reading separately and reported against the reading it matches best (schema recall
first, then data F1, then schema precision). Every result column of a reading is annotated with

* a **relevance level**: `REQUIRED` (named `PREFERRED` in the code) or `ACCEPTABLE`. Only required
  concepts drive recall; acceptable columns are wrapped in `OPTIONAL` in the SPARQL so that rows without
  them are not dropped;
* a **semantic concept** (`semantic_concept`): the thing the question asks about, in the terms of the
  question. Several columns can realise one concept (an identifier and a label); producing either one
  counts as addressing the concept.

Example (`BASE06`, transit):

```python
AdaptiveGroundTruth(
    query_id="BASE06", query_set="BASE", dataset="TRN",
    query="What is the total number of stop events and average stop events per journey for each line?",
    sparql_queries=[...],                       # four readings: with/without terminal stops, with/without implicit types
    columns=[
        ColumnDef("route", RelevanceLevel.PREFERRED, semantic_concept="route"),
        ColumnDef("routeName", RelevanceLevel.PREFERRED, semantic_concept="route"),
        ColumnDef("totalDepartures", RelevanceLevel.PREFERRED, semantic_concept="total", is_measurement=True),
        ...
    ],
)
```

The readings were executed against the endpoints; their results are cached in
`data/cache/ground_truth_full.json` so that runs can be re-evaluated without the endpoints.

## Questions

Datasets: EDU (university), TRN (public transit), NRG (oil and gas), BSBM (e-commerce), LCA (life-cycle
assessment). The substitutions of the SYN and TYPO variants are listed in `corpus.csv`.

| ID | Condition | Dataset | Readings | Question |
|---|---|---|---|---|
| BASE01 | BASE | EDU | 2 | Show me 5000 examples of learners enrolled in courses instructed by educators employed at their own division. |
| BASE03 | BASE | LCA | 1 | List activities with their input and output flows. |
| BASE04 | BASE | TRN | 4 | Show me 1000 examples of which lines have stop events at which stations. |
| BASE05 | BASE | EDU | 2 | How many learners in each division are enrolled in courses, and what courses are they enrolled in? |
| BASE06 | BASE | TRN | 4 | What is the total number of stop events and average stop events per journey for each line? |
| BASE08 | BASE | TRN | 10 | Show the top 10 stations with the most journeys. |
| BASE09 | BASE | EDU | 1 | Which educators who instruct advanced modules have no mentees? |
| BASE10 | BASE | BSBM | 2 | What are the cheapest and most expensive products? |
| BASE11 | BASE | NRG | 2 | Which deposits have more than 50 million in extracted crude total? |
| BASE12 | BASE | EDU | 2 | List all basic modules and advanced modules with their instructors and division. |
| BASE13 | BASE | NRG | 2 | Show the condition and active deposit operator of the TROLL deposit. |
| BASE14 | BASE | EDU | 2 | Show courses instructed at University0. |
| BASE16 | BASE | NRG | 2 | Which deposits have Statoil as their active deposit operator? |
| BASE18 | BASE | NRG | 2 | Find boreholes with a total drill depth greater than 3000 meters. |
| BASE19 | BASE | TRN | 4 | Show journeys with leave times between 00:00:00 and 01:00:00. |
| BASE20 | BASE | NRG | 4 | For each active deposit operator, list all the deposits they operate. |
| BASE21 | BASE | NRG | 2 | Which deposits are operated by Det norske oljeselskap ASA? |
| SYN01 | SYN | EDU | 2 | Show me 5000 cases of students taking classes taught by professors from their own department. |
| SYN03 | SYN | LCA | 1 | Show unit processes with their exchanges. |
| SYN04 | SYN | TRN | 4 | Show me 1000 cases of which transit routes pass through which platforms. |
| SYN05 | SYN | EDU | 2 | Per department, how many people are registered for classes, and which ones? |
| SYN06 | SYN | TRN | 4 | For each transit route, how many times does it halt altogether and what is the mean per trip? |
| SYN08 | SYN | TRN | 10 | Which 10 places in the transit network are served by the most trips? |
| SYN09 | SYN | EDU | 1 | Which faculty teaching upper-level classes have nobody under their supervision? |
| SYN10 | SYN | BSBM | 2 | Which items are the least expensive and the costliest? |
| SYN11 | SYN | NRG | 2 | Where has more than 50000000 in oil been pumped out altogether? |
| SYN12 | SYN | EDU | 2 | Show every undergraduate and graduate class alongside who teaches it and which department it falls under. |
| SYN13 | SYN | NRG | 2 | Show the status and managing company of the Troll oil field. |
| SYN14 | SYN | EDU | 2 | Show classes offered at Uni 0. |
| SYN16 | SYN | NRG | 2 | Which fields are currently operated by Equinor? |
| SYN18 | SYN | NRG | 2 | Which drill holes go deeper than 3 kilometers? |
| SYN19 | SYN | TRN | 4 | Show transit trips departing between midnight and 1 AM. |
| SYN20 | SYN | NRG | 4 | For each oil company, which production sites do they currently manage? |
| SYN21 | SYN | NRG | 4 | Which fields does Aker BP currently manage? |
| TYPO01 | TYPO | EDU | 2 | Show me 5000 exampels of learnres enrolld in coureses instructd by educatros employd at their own divison. |
| TYPO03 | TYPO | LCA | 1 | List activites with thier input and outptu flows. |
| TYPO04 | TYPO | TRN | 4 | Show me 1000 exampels of wich lines have stopp events at wich statinos. |
| TYPO05 | TYPO | EDU | 2 | How manny learnres in each divison are enrolld in coureses, and what coureses are they enrolld in? |
| TYPO06 | TYPO | TRN | 4 | What is the totel number of stopp events and averge stopp events per journy for each lin? |
| TYPO08 | TYPO | TRN | 10 | Show the top 10 statinos with the mostt journyes. |
| TYPO09 | TYPO | EDU | 1 | Which educatros who instrcut advnaced moduels have no menteees? |
| TYPO10 | TYPO | BSBM | 2 | What are the cheepest and most expensiv prodcuts? |
| TYPO11 | TYPO | NRG | 2 | Which deposists have more than fivty million in extrcated crude totel? |
| TYPO12 | TYPO | EDU | 2 | List all baisc moduels and advnaced moduels with their instructros and divison. |
| TYPO13 | TYPO | NRG | 2 | Show the conditon and actve deposit opertor of the TROL deposite. |
| TYPO14 | TYPO | EDU | 2 | Show coureses instructd at Univeristy0. |
| TYPO16 | TYPO | NRG | 2 | Which deposists have Statoil as their actve deposit opertor? |
| TYPO18 | TYPO | NRG | 2 | Find boreholse with a totel drill deptth greater than 3000 meeters. |
| TYPO19 | TYPO | TRN | 4 | Show journyes with leav tiems between 00:00:000 and 01:00:000. |
| TYPO20 | TYPO | NRG | 4 | For each actve deposit opertor, lsit all the deposists they oprate. |
| TYPO21 | TYPO | NRG | 2 | Which deposists are operatd by Det norske oljeslskap ASA? |
| UNDER01 | UNDER | BSBM | 2 | Show top products. |
| UNDER03 | UNDER | NRG | 2 | Get recent production data. |
| UNDER04 | UNDER | EDU | 2 | Find large departments. |
| UNDER06 | UNDER | LCA | 1 | Find significant emissions. |
| UNDER08 | UNDER | BSBM | 2 | Show affordable products. |

