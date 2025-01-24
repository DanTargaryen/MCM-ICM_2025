import pandas as pd

df = pd.read_csv("summerOly_athletes.csv")

df = df[df['Year'].isin([2020, 2024])]

df_gold = df[df['Medal'] == 'Gold']

medals = {}

for _, row in df_gold.iterrows():
    country_name = f"{row['NOC']}-{row['Name']}"
    sport = row['Sport']

    if country_name not in medals:
        medals[country_name] = {}

    if sport not in medals[country_name]:
        medals[country_name][sport] = 0

    medals[country_name][sport] += 0.5

athletes = sorted(medals.keys())

sports = sorted(set(row['Sport'] for _, row in df_gold.iterrows()))

columns = sports

result_df = pd.DataFrame(index=athletes, columns=columns)

for athlete in athletes:
    for sport in columns:
        result_df.at[athlete, sport] = medals.get(athlete, {}).get(sport, 0)

result_df.to_csv("result_with_gold_medals_transposed.csv", index=True)

print("CSV文件已生成")
