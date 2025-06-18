import openai
import pandas as pd
import ast

openai.api_key = ''

print("Loading data...")
path = 'data/'
mAmbience = pd.read_parquet(path + 'ch2025_mAmbience.parquet')

mAmbience['timestamp'] = pd.to_datetime(mAmbience['timestamp'])

category_set = set()

for _, row in mAmbience.iterrows():
    if isinstance(row['m_ambience'], str):
        try:
            m_ambience_list = ast.literal_eval(row['m_ambience'])
            if isinstance(m_ambience_list, list):
                for item in m_ambience_list:
                    if isinstance(item, (list, np.ndarray)) and len(item) == 2:
                        category = item[0].strip()
                        category_set.add(category)
        except:
            continue

    elif isinstance(row['m_ambience'], (list, np.ndarray)):
        for item in row['m_ambience']:
            if isinstance(item, (list, np.ndarray)) and len(item) == 2:
                category = item[0].strip()
                category_set.add(category)

category_list = list(category_set)
total_categories = len(category_list)
print(f"Total categories: {total_categories}")

category_data = [{"category": category, "weight": -1} for category in category_list]

batch_size = 100
category_batches = [category_data[i:i + batch_size] for i in range(0, len(category_data), batch_size)]

summary_prompt = """
Each category should be evaluated based on its impact on sleep quality, physical fatigue, stress level, and adherence to sleep guidelines (total sleep time, sleep efficiency, sleep onset latency).
Please assign a weight between 0 and 1 for each category, reflecting how much it influences these elements.

The output format should be as follows:
- Category name: weight
For example:
Singing bowl: 0.8
Scrape: 0.6
Crushing: 0.7
Rustle: 0.5
Marimba, xylophone: 0.6
...
Please ensure that each category is listed on a new line and that no numbers or additional text are included.
"""

for batch in category_batches:
    categories_text = "\n".join([f"{item['category']}" for item in batch])
    prompt = f"""
    {summary_prompt}

    Categories:
    {categories_text}
    """

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": "You are a helpful assistant."},
                  {"role": "user", "content": prompt}],
        max_tokens=1500,
        temperature=0.5
    )

    weights_text = response['choices'][0]['message']['content'].strip()

    if weights_text:
        for line in weights_text.split("\n"):
            if ':' in line:
                try:
                    category, weight_value = line.split(":")
                    category = category.strip()
                    category = category.lstrip('- ')
                    weight_value = float(weight_value.strip())

                    for item in batch:
                        if item["category"] == category:
                            item["weight"] = weight_value
                except ValueError:
                    continue

missing_categories = [item for item in category_data if item["weight"] == -1]

if missing_categories:
    print(f"Missing categories: {[item['category'] for item in missing_categories]}")
    missing_batches = [missing_categories[i:i + batch_size] for i in range(0, len(missing_categories), batch_size)]
    for batch in missing_batches:
        categories_text = "\n".join([f"{item['category']}" for item in batch])
        prompt = f"""
        {summary_prompt}

        Categories:
        {categories_text}
        """

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": "You are a helpful assistant."},
                      {"role": "user", "content": prompt}],
            max_tokens=1500,
            temperature=0.5
        )

        weights_text = response['choices'][0]['message']['content'].strip()
        print(f"Response received for missing categories batch:")
        print(weights_text)

        if weights_text:
            for line in weights_text.split("\n"):
                if ':' in line:
                    try:
                        category, weight_value = line.split(":")
                        category = category.strip()
                        category = category.lstrip('- ')
                        weight_value = float(weight_value.strip())

                        for item in batch:
                            if item["category"] == category:
                                item["weight"] = weight_value
                    except ValueError:
                        continue

df_weights = pd.DataFrame(category_data)

output_path = path + 'weight_final.csv'
df_weights.to_csv(output_path, index=False)

print(f"Category weights have been saved to {output_path}")
