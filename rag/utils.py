mapper_partis = {'Groupe Renew Europe (Renaissance, MoDem, Parti radical, Horizons, Ecologie au centre) ': 'Groupe Renew Europe',
 "Groupe de l'Alliance Progressiste des Socialistes et Démocrates au Parlement européen (Parti Socialiste) ": "Groupe de l'Alliance Progressiste des Socialistes et Démocrates au Parlement européen",
 'Groupe des Conservateurs et Réformistes européens (Reconquête) ': 'Groupe des Conservateurs et Réformistes européens',
 'Groupe des Verts/Alliance libre européenne (EELV, Femu a Corsica, Union Démocratique Bretonne) ': 'Groupe des Verts/Alliance libre européenne',
 'Groupe du Parti populaire européen (Démocrates-Chrétiens) (Les Républicains, les Centristes) ': 'Groupe du Parti populaire européen (Démocrates-Chrétiens)',
 'Groupe «Identité et démocratie» (Rassemblement National, La Droite Populaire ) ': 'Groupe «Identité et démocratie»',
 'Le groupe de la gauche au Parlement européen - GUE/NGL (LFI, Gauche républicaine et socialiste) ': 'Le groupe de la gauche au Parlement européen - GUE/NGL',
 }

#'Non-inscrits (Non-inscrits) ': 'Non-inscrits'


assistant_mapper_party = {'Groupe Renew Europe' : "ValeRAG Hayer",
  "Groupe de l'Alliance Progressiste des Socialistes et Démocrates au Parlement européen" : 'Glucksllman',
 'Groupe des Conservateurs et Réformistes européens' : "NicolAI Bay",
 'Groupe des Verts/Alliance libre européenne' : 'Marie ToussAInt',
  'Groupe du Parti populaire européen (Démocrates-Chrétiens)' : 'Nadine Moranocode', 
  'Groupe «Identité et démocratie»' : 'Jordan BardeLLAMA', 
  'Le groupe de la gauche au Parlement européen - GUE/NGL' : 'Manon AubRAG',
 }


def get_response(retrieval_chain ,  user_input: str, party : str):
    """get_response 

    Args:
        retrieval_chain (_type_): _description_
        user_input (str): _description_

    Yields:
        _type_: _description_
    """
    response_stream= retrieval_chain.stream(
        {'input': user_input, 'party' : party}
    ) 
    for chunk in response_stream:
        content=chunk.get("answer","")
        yield content

def stream_str(sequence :str) : 
    for word in sequence.split(" ") :
        yield word + " "


def generate_response_by_groups(user_query, partis, retrieval_chain, retriever) : 
    
    summaries = {}
    for parti in partis : 
        result  = get_response(retrieval_chain, user_query)
        context = retriever.invoke(user_query)

        if context != [] : 
            summaries[parti] = {"result" : result,
                                "context" : context}

    return summaries 
            

