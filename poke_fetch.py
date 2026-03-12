import requests
import time
import json
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

'''
Gets data set from PokeAPI and converts data to csv file for easy access.
'''

def fetch_pokemon(pokemon_id: int) -> dict:
    poke_attributes = {}

    poke_data = requests.get(f'https://pokeapi.co/api/v2/pokemon/{pokemon_id}')
    data = poke_data.json()

    poke_attributes['id'] = pokemon_id
    poke_attributes['name'] = data['name']
    if len(data['types']) > 1:
        poke_attributes['type1'] = data['types'][0]['type']['name']
        poke_attributes['type2'] = data['types'][1]['type']['name']
    else:
        poke_attributes['type1'] = data['types'][0]['type']['name']
        poke_attributes['type2'] = None
    
    for stat in data['stats']:
        stat_name = stat['stat']['name']
        poke_attributes[stat_name] = stat['base_stat']
    
    species_data = requests.get(data['species']['url'])
    evo_chain_url = species_data.json()['evolution_chain']['url']
    evo_response = requests.get(evo_chain_url)
    evo_data = evo_response.json()
    
    final_evo = is_fully_evolved(evo_data['chain'], data['name'])
    poke_attributes['final_evo'] = final_evo if final_evo is not None else False

    return poke_attributes

def is_fully_evolved(chain, pokemon_name):
    current_mon = chain['species']['name']

    if current_mon == pokemon_name:
        return len(chain['evolves_to']) == 0
    
    for evolution in chain['evolves_to']:
        res = is_fully_evolved(evolution, pokemon_name)
        if res is not None:
            return res
        
    return None


if __name__ == "__main__":
    all_pokemon = [0] * 151
    for i in range(1, 152):
        time.sleep(0.1)
        pokemon = fetch_pokemon(i)
        all_pokemon[i - 1] = pokemon
        print("Fetching Pokemon " + str(i))
    df = pd.DataFrame(all_pokemon)
    df.to_csv('Gen1_data.csv', index=False)
    
    



    
    
