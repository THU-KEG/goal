import os, sys, json, re
import argparse
import tqdm
import numpy as np
import requests
import pandas as pd
import pickle
from SPARQLWrapper import SPARQLWrapper, JSON
import urllib

DEFAULT_BLINK = "http://0.0.0.0:9271/blink_entity"


PROXIES = {
           'http': 'http://127.0.0.1:8001',
           'https': 'http://127.0.0.1:8001',
        }

proxy_support = urllib.request.ProxyHandler(PROXIES)
opener = urllib.request.build_opener(proxy_support)
urllib.request.install_opener(opener)




def link_mentions_to_wikipedia(mentions, blink_interface, save_linked_entities="linked_entities.json"):
    """ Link entities mentions to Wikipedia entities based on BLINK.
      Args:
        mentions: a dict with mention name as keys and the list of mention context sentences as values.
      Return:
        A dict with mention name as keys, each value including the entity title, the url to Wikipedia page, the Wikida item id.
        Note that the final entity would be selected by the **majority voting** over all contexts
    """
    if not os.path.exists(save_linked_entities):
    
        result = {} # [mention_name: [(entity_title, url, WikidataID)] ], each value representing a list according to different mention context
        cache_props = {}
        for men_name in tqdm.tqdm(mentions, "linking mentions to wikipedia and wikidata ..."):
            cand_entities = {} # over contexts
            for context in mentions[men_name]:
                # try:
                men_poss = context.find(men_name)
                l_context = context[:men_poss] if men_poss!=-1 else ""
                r_context = context[men_poss+len(men_name):] if men_poss!=-1 else ""
                response = requests.post(blink_interface, json={"mention": men_name, "context_left":l_context, "context_right":r_context})
                # if response.content.decode("utf-8") is "None":
                #     continue
                try:
                    predictions = json.loads(response.content.decode("utf-8"))["predictions"]
                except:
                    print("Invalid return from BLINK for mention: %s " % men_name)
                    continue
                e_title, e_url = predictions[0][0]
                e_id = re.match(r".*?curid=(.*?)$", e_url).group(1)
                
                # get wikidata item id
                if e_id in cache_props:
                    e_wikipedia_props = cache_props[e_id]
                else:
                    n_try = 10
                    while True and n_try>0:
                        try:
                            e_wikipedia_props = requests.get(f"https://en.wikipedia.org/w/api.php?action=query&prop=pageprops&pageids={e_id}&inprop=url&format=json", proxies=PROXIES)
                            if e_id not in cache_props:
                                cache_props[e_id] = e_wikipedia_props
                            n_try = 0
                        except:
                            n_try -= 1
                            print("Retrying %d-th to get wikipedia page info for curid= %s" % (10-n_try, e_id))
                try:
                    e_wikipedia_props = json.loads(e_wikipedia_props.content.decode("utf-8"))
                    e_wikidata_qid = e_wikipedia_props["query"]["pages"][e_id]["pageprops"]["wikibase_item"]
                    e_wikidata_qid = e_wikidata_qid[1:]
                except KeyError:
                    print("There is a KeyError of 'pageprops' for mention name: %s " % men_name)
                    continue
                    

                cand_entities[e_wikidata_qid] = cand_entities.get(e_wikidata_qid, []) + [(e_title, e_url, e_wikidata_qid)]
                # except:
                #     print("Error happened on mention: {}, and context: {}".format(men_name, context))
            # Majority voting
            if len(cand_entities) > 0:
                final_entity = max(cand_entities.values(), key=lambda x:len(x))[0]
                result[men_name] = final_entity
            else:
                print("There is not linked entity in wikipedia for mention {}".format(men_name))
        with open(save_linked_entities, "w") as f:
            json.dump(result, f)
    else:
        with open(save_linked_entities, "r") as f:
            result = json.load(f)
            
    return result


def get_1hop_kgs_wikidata(linked_entities, save_kgs="kgs.xlsx"):
    """ Taking the given entity as center node, get its 1-hop knowledge graph.
      Args:
        linked_entities: A dict with mention name as keys, each value including the entity title, the url to Wikipedia page, the Wikida item id.
      Return:
        A list of dicts each representing a kg with column name as keys.
    """
    if not os.path.exists(save_kgs):
        endpoint_url = "https://query.wikidata.org/sparql"

        result_kgs = [] # [(mention_name, kg) ]

        for men_name in tqdm.tqdm(linked_entities, "Obtaining 1-hop kgs ..."):
            try:
                e_id = linked_entities[men_name][-1]
                query = """
                    SELECT ?p ?wdLabel ?ps_ ?ps_Label ?wdpqLabel ?pq_Label {
                        VALUES (?item) {(wd:Q%s)}

                        ?item ?p ?statement .
                        ?statement ?ps ?ps_ .

                        ?wd wikibase:claim ?p.
                        ?wd wikibase:statementProperty ?ps.

                        OPTIONAL {
                        ?statement ?pq ?pq_ .
                        ?wdpq wikibase:qualifier ?pq .
                        }

                        SERVICE wikibase:label { bd:serviceParam wikibase:language "en" }
                    } ORDER BY ?wd ?statement ?ps_
                """ % e_id

                user_agent = "WDQS-example Python/%s.%s" % (sys.version_info[0], sys.version_info[1])
                # TODO adjust user agent; see https://w.wiki/CX6
                sparql = SPARQLWrapper(endpoint_url, agent=user_agent)
                sparql.setQuery(query)
                sparql.setReturnFormat(JSON)
                response = sparql.query().convert()

                df = pd.json_normalize(response["results"]["bindings"])
                df_filtered = df[df.filter(like='value').columns] # keep the columns that have the value
                df_filtered = df_filtered.rename(columns={"p.value": "relation_uri", "wdLabel.value": "relation_label", "ps_.value": "object_uri", "ps_Label.value": "object_label", "wdpqLabel.value": "qualifier_label", "pq_Label.value": "qualifier_value"})

                result_kgs.append((e_id, df_filtered))
            except:
                print("Failed to get KG from Wikidata for mention nane: %s" % men_name)

        # save
        with pd.ExcelWriter(save_kgs) as writer:
            for e_id, df in result_kgs:
                df.to_excel(writer, sheet_name=e_id, index=False)
    else:
        result_kgs = []
        df_sheets = pd.read_excel(save_kgs, sheet_name=None, engine="openpyxl")
        for sh_name in df_sheets:
            df = df_sheets[sh_name]
            result_kgs.append((sh_name, df))
    
    return result_kgs



def merge_entities_kgs(kgs, save_kg="kg.ttl"):
    """ 
    """
    pass



def get_mentions(game_commentary, tags=["team", "player"], save_mentions="mentions.json"):
    """ Get all entity mentions from a given commentary json file.
    """
    if not os.path.exists(save_mentions):
        mentions = {} # {"mention_name": set(contexts)}
        with open(game_commentary) as f:
            comm_data = json.load(f)
        for X in comm_data:
            for tag in tags:
                match = re.match(r".*?\[{}\]\s?(.*?)\s?\[{}\].*?".format(tag,tag), str(X["entity"]))
                if match:
                    men_name = match.group(1)
                    if men_name == "":
                        continue

                    if men_name not in mentions:
                        mentions[men_name] = [X["commentary"]]
                    elif X["commentary"] not in mentions[men_name]:
                        mentions[men_name].append(X["commentary"])
        with open(save_mentions, "w") as f:
            json.dump(mentions, f)
    else:
        with open(save_mentions, "r") as f:
            mentions = json.load(f)
    print("Total number of distinct mentions names: %d ." % len(mentions))
    return mentions
                



def generate(commentaries_dir, blink_interface, save_dir):
    """ Generate KG and save.
    """
    for f_name in tqdm.tqdm(os.listdir(commentaries_dir), desc="Overall progress"):
        print("Processing the video: {}".format(f_name))
        
        kg_dir = os.path.join(save_dir, f_name.replace(".json", ""))
        os.makedirs(kg_dir, exist_ok=True)
        
        # Get entities mentions
        f_path = os.path.join(commentaries_dir, f_name)
        mentions = get_mentions(f_path, save_mentions=os.path.join(kg_dir, "mentions.json"))
        
        # Get linked entities from Wikipedia based on BLINK
        linked_entities = link_mentions_to_wikipedia(mentions, blink_interface, save_linked_entities=os.path.join(kg_dir, "linked_entities.json"))
        
        # Get 1-hop KG with current entity as center point
        # entities_wikidata_ids = [linked_entities[men_name][-1] for men_name in linked_entities]
        entities_kgs = get_1hop_kgs_wikidata(linked_entities, save_kgs=os.path.join(kg_dir, "1-hop_kgs.xlsx"))
        
        # Merge all entities-centered sub-kgs and save to ttl
        merge_entities_kgs(entities_kgs, save_kg=os.path.join(kg_dir, "kgs.ttl"))




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--blink_interface", type=str, default=DEFAULT_BLINK)
    parser.add_argument("--commentary_dir", type=str, default="/data/qiji/DATA/soccernet/commentaries")
    parser.add_argument("--kgs_dir", type=str, default="/data/qiji/DATA/soccernet/kgs")
    args = parser.parse_args()

    generate(args.commentary_dir, args.blink_interface, args.kgs_dir)