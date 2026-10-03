# Qualitative samples (dev, beam search k=4)

Five correct and five wrong examples, chosen with a fixed random seed. SQL uses the real column names.

## Correct

**Dev example 5561**

- Question: What competition has a score greater than 30, a draw less than 5, and a loss larger than 10?
- Columns: Season, Competition, Matches, Draw, Lost, Points
- Gold SQL: `SELECT Competition FROM table WHERE Points > 30 AND Draw < 5 AND Lost > 10`
- Our SQL: `SELECT Competition FROM table WHERE Points > 30 AND Draw < 5 AND Lost > 10`

**Dev example 7264**

- Question: What is the total average for Rank entries where the Lane listed is smaller than 4 and the Nationality listed is San Marino?
- Columns: Rank, Lane, Athlete, Nationality, Time, React
- Gold SQL: `SELECT AVG(Rank) FROM table WHERE Lane < 4 AND Nationality = 'san marino'`
- Our SQL: `SELECT AVG(Rank) FROM table WHERE Lane < 4 AND Nationality = 'san marino'`

**Dev example 8049**

- Question: Name the score which has record of 73-83
- Columns: Date, Opponent, Score, Loss, Record
- Gold SQL: `SELECT Score FROM table WHERE Record = '73-83'`
- Our SQL: `SELECT Score FROM table WHERE Record = '73-83'`

**Dev example 2053**

- Question: How many years have a weeks at #1 value of exactly 8?
- Columns: Artist, Country, Number-one single(s), Year, Weeks at #1, Straight to #1 ?
- Gold SQL: `SELECT COUNT(Year) FROM table WHERE Weeks at #1 = 8`
- Our SQL: `SELECT COUNT(Year) FROM table WHERE Weeks at #1 = 8`

**Dev example 6536**

- Question: Which Torque has a Model of s63 amg ('01)?
- Columns: Model, Engine, Cyl., Power, Torque
- Gold SQL: `SELECT Torque FROM table WHERE Model = 's63 amg (''01)'`
- Our SQL: `SELECT Torque FROM table WHERE Model = 's63 amg (''01)'`

## Wrong

**Dev example 1246**

- Question: What is the largest n value for 55.6% r1b1c4 (r-v69)?
- Columns: Region, Population, Country, Language, N, Total%, R1b1c (R-V88), R1b1a2 (R-M269), R1b1c* (R-V88*), R1b1c4 (R-V69)
- Gold SQL: `SELECT MAX(N) FROM table WHERE R1b1c4 (R-V69) = '55.6%'`
- Our SQL: `SELECT MAX(N) FROM table WHERE R1b1c* (R-V88*) = '55.6%'`
- Failure: **wrong column in WHERE**

**Dev example 8242**

- Question: What sum of Losses has Year greater than 1972, and Competition of nswrfl, and Draws 0, and Wins 16?
- Columns: Year, Competition, Wins, Draws, Loses
- Gold SQL: `SELECT SUM(Loses) FROM table WHERE Year > 1972 AND Competition = 'nswrfl' AND Draws = 0 AND Wins = 16`
- Our SQL: `SELECT SUM(Loses) FROM table WHERE Year > 1972 AND Competition = 'nswrfl' AND Wins = 16 AND Wins = 16`
- Failure: **missing condition**

**Dev example 5158**

- Question: What's the average amount of points for "in and out of love" with a draw over 8?
- Columns: Draw, Artist, Song, Points, Place
- Gold SQL: `SELECT AVG(Points) FROM table WHERE Song = '"in and out of love"' AND Draw > 8`
- Our SQL: `SELECT AVG(Points) FROM table WHERE Draw > 8 AND Artist = 'love"' AND Song = 'love"'`
- Failure: **extra condition**

**Dev example 625**

- Question: What is the sexual abuse rate where the conflict is the Burundi Civil War?
- Columns: Conflict, United Nations Mission, Sexual abuse 1, Murder 2, Extortion/Theft 3
- Gold SQL: `SELECT MAX(Sexual abuse 1) FROM table WHERE Conflict = 'Burundi Civil War'`
- Our SQL: `SELECT Sexual abuse 1 FROM table WHERE Conflict = 'burundi civil war'`
- Failure: **wrong aggregation**

**Dev example 5235**

- Question: Name the AEDT Time which has an Away team of collingwood?
- Columns: Home team, Home team score, Away team, Away team score, Ground, Crowd, Date, Local Time, AEDT Time
- Gold SQL: `SELECT AEDT Time FROM table WHERE Away team = 'collingwood'`
- Our SQL: `SELECT Local Time FROM table WHERE Away team = 'collingwood'`
- Failure: **wrong select column**
