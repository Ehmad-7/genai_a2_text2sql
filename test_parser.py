from decode import parse_query

# basic: select a column with one condition
assert parse_query("select <c2> where <c0> = terrence ross") == \
    {"sel": 2, "agg": 0, "conds": [[0, 0, "terrence ross"]]}

# aggregate, no where
assert parse_query("select count <c3>") == {"sel": 3, "agg": 3, "conds": []}

# aggregate with a condition
assert parse_query("select max <c1> where <c0> = x") == \
    {"sel": 1, "agg": 1, "conds": [[0, 0, "x"]]}

# two conditions with > and <
assert parse_query("select <c1> where <c0> > 5 and <c2> < 10") == \
    {"sel": 1, "agg": 0, "conds": [[0, 1, "5"], [2, 2, "10"]]}

# the word "and" inside a value must NOT split the condition
assert parse_query("select <c1> where <c0> = tom and jerry") == \
    {"sel": 1, "agg": 0, "conds": [[0, 0, "tom and jerry"]]}

# the word "where" inside a value
assert parse_query("select <c1> where <c0> = where is it") == \
    {"sel": 1, "agg": 0, "conds": [[0, 0, "where is it"]]}

# "and" inside a value, followed by a real second condition
assert parse_query("select <c1> where <c0> = tom and jerry and <c3> = 7") == \
    {"sel": 1, "agg": 0, "conds": [[0, 0, "tom and jerry"], [3, 0, "7"]]}

# malformed strings return None, and never raise
for bad in ["hello world", "select where", "select <c1> where",
            "select <c1> foo", "", "<c1> where <c0> = x"]:
    assert parse_query(bad) is None, bad

print("parser tests passed")