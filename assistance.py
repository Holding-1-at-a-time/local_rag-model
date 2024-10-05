import chromadb
import ollama
import psycopg
from psycopg.rows import dict_row
from colorama import Fore
import os
import ast
from tqdm import tqdm
from dotenv import load_dotenv

# The `load_dotenv()` function is used to load environment variables from a .env file into the
# script's environment. This allows the script to access sensitive information such as database
# credentials without hardcoding them directly into the code.

load_dotenv()

# The above code snippet appears to be setting up a Python script that involves creating a client for
# a database (referred to as `chromadb`), defining a system prompt message, initializing a
# conversation list (`convo`), and setting up database connection parameters (`DB_PARAMS`).

client = chromadb.Client()

system_prompt = (
    "You will never assume anything, and will never guess on any answers you generate. "
    "You are an AI assistant equipped with advanced capabilities, including the use of embeddings, Chain of Thought (CoT) reasoning, and self-reflection."
    "For every prompt from the user: Analyze: Review any relevant embeddings from previous conversations to inform your responses. "
    "Integrate Context: If the context provided by these embeddings is useful and pertinent, incorporate it into your reply. "
    "CoT Reasoning: Employ Chain of Thought reasoning to logically navigate through complex queries, ensuring your thought process is clear and structured. "
    "Self-Reflection: After providing a response, assess its effectiveness and consider whether additional clarification or detail could enhance user understanding. "
    "Respond Appropriately: If the context is not relevant, respond directly to the user, focusing on delivering precise and professional assistance as an intelligent AI assistant. "
)
# The code snippet you provided is initializing a conversation list named `convo` and setting up
# database connection parameters in a dictionary named `DB_PARAMS`. Here's what each part is doing:

convo = [{"role": "system", "content": system_prompt}]

DB_PARAMS = {
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT"),
}


def connect_db() -> psycopg.Connection:
    """
    Connect to the PostgreSQL database.

    Returns:
        psycopg.Connection: An active connection to the database.
    """
    print("Connecting to database with params:", DB_PARAMS)
    try:
        conn = psycopg.connect(**DB_PARAMS)
        print("Connected to database")
        return conn
    except psycopg.Error as e:
        print("Error connecting to database:", e)
        raise


def fetch_conversations():
    print("Fetching conversations from database")
    conn = connect_db()
    with conn.cursor(row_factory=dict_row) as cursor:
        cursor.execute("SELECT * FROM conversations")
        conversations = cursor.fetchall()
        print(f"Fetched {len(conversations)} conversations")

    conn.close()
    return conversations


def store_conversations(prompt, response):
    conn = connect_db()
    with conn.cursor() as cursor:
        cursor.execute(
            "INSERT INTO conversations (timestamp, prompt, response) VALUES (CURRENT_TIMESTAMP, %s, %s)",
            (prompt, response),
        )
        conn.commit()
    print(f"Stored conversation with prompt: {prompt}")
    conn.close()


def remove_last_conversation():
    conn = connect_db()
    with conn.cursor() as cursor:
        cursor.execute(
            "DELETE FROM conversations WHERE id = (SELECT MAX(id) FROM conversations)"
        )
        cursor.commit()
    conn.close()


def stream_response(prompt):
    response = ""
    stream = ollama.chat(model="llama3.1:8b", message=convo, stream=True)
    print(Fore.CYAN + "\nASSISTANT:")

    for chunk in stream:
        content = chunk["message"]["content"]
        response += content
        print(content, end="", flush=True)

    print("\n")
    store_conversations(prompt=prompt, response=response)
    convo.append({"role": "assistant", "content": response})


def create_vector_db(conversations):
    vector_db_name = "conversations"

    try:
        client.delete_collection(name=vector_db_name)
    except Exception as e:
        print(f"Error deleting collection: {e}")
        # or log the error, or take some other appropriate action
    vector_db = client.create_collection(name=vector_db_name)

    for c in conversations:
        print(f"Adding conversation with id: {c['id']} to vector db")
        serialized_convo = f'prompt: {c['prompt']} response: {c['response']}'

        response = ollama.embeddings(model="nomic-embed-text", prompt=serialized_convo)
        embedding = response["embedding"]

        vector_db.add(
            ids=[str(c["id"])],
            embeddings=[embedding],
            documents=[serialized_convo],
        )


def retrieve_embeddings(queries, results_per_query=3):
    embeddings = set()

    for query in tqdm(queries, desc="process queries to vector database"):
        response = ollama.embeddings(model="nomic-embed-text", prompt=query)
        query_embedding = response["embedding"]

        vector_db = client.get_collection(name="conversations")
        print(f"Querying vector db with query: {query}")
        results = vector_db.query(
            query_embeddings=[query_embedding], n_results=results_per_query
        )
        best_embeddings = results["documents"][0]

        for best in best_embeddings:
            print(f"Found embedding: {best}")
            if best not in embeddings and "yes" in classify_embedding(
                query=query, context=best
            ):
                embeddings.add(best)

    return embeddings


def create_queries(prompt):
    query_msg = (
        "You are a first principal reasoning and logical search query AI agent. "
        "You will never assume anything, and will never guess on any answers you generate. "
        "Your list of search queries will be ran on an embeddings database of all your conversations "
        "you have ever had with the user. With first principal reasoning and logic create a Python list of queries to"
        "search the embeddings database for any data that would be relevant and necessary to have access to in "
        "order to correctly respond to the user prompt. Your response must be in a Python list with no syntax errors. "
        "Do not explain anything and do not ever generate anything but a perfect syntax Python list. "
        "Please use the information from your most recent conversation to generate your search queries. "
    )

    query_convo = [
        {"role": "system", "content": query_msg},
        {
            "role": "user",
            "content": "Please generate a list of Python list of queries to search the embeddings database for any data that would be relevant and necessary to have access to in order to correctly respond to the user prompt.",
        },
        {
            "role": "assistant",
            "content": '["what is the users name?", "what is the users age?", "what is the users email?", "what is the users phone number?"]',
        },
        {
            "role": "user",
            "content": "How can I utilize NextJs App Router in my multitenancy SaaS application I want to create?",
        },
        {
            "role": "assistant",
            "content": '["you can use NextJs app router to create a multitenancy SaaS application", "you can use NextJs app router to create a multitenancy SaaS application by using the routing system", "you can also possible to use NextJs app router to create a multitenancy SaaS application by using the routing system, for instance parallel routing, nest routing, or dynamic routing."]',
        },
        {"role": "user", "content": prompt},
    ]

    response = ollama.chat(model="llama3.1:8b", message=query_convo)
    print(
        Fore.YELLOW + f'\nVector database queries: {response["message"]["content"]} \n'
    )

    try:
        return ast.literal_eval(response["message"]["content"])
    except Exception:
        return [prompt]


def classify_embedding(query, context):
    classify_msg = (
        "You are an embeddings classification AI agent. your input will be a prompt and one embedded chunk of text. "
        "You will not respond as an AI assistant. You will only answer with a yes or no. "
        "Determine whether the context contains data that is directly is related to the search query. "
        'If the context contains data that is directly is related to the search query, respond with "yes", '
        'if it is anything else but directly related respond with "no. Do not respond "yes" unless the context is highly relevant to '
        "the search query. "
    )

    classify_convo = (
        {"role": "system", "content": classify_msg},
        {
            "role": "user",
            "content": f"SEARCH QUERY: What is the users name? \n\nEMBEDDED CONTEXT: You are Robert Romero. How can I assist you?",
        },  # noqa: F541
        {"role": "assistant", "content": "Yes"},
        {
            "role": "user",
            "content": f"SEARCH QUERY: Llama3 voice assistant \n\nEMBEDDED CONTEXT: I am a voice assistant. How can I assist you?",
        },  # noqa: F541
        {"role": "assistant", "content": "No"},
        {
            "role": "user",
            "content": f"SEARCH QUERY: {query} \n\nEMBEDDED CONTEXT: {context}",
        },
    )

    response = ollama.chat(model="llama3.1:8b", message=classify_convo)

    return response["message"]["content"].strip().lower()


def recall(prompt):
    queries = create_queries(prompt=prompt)
    embeddings = retrieve_embeddings(queries=queries)
    convo.append(
        {
            "role": "user",
            "content": f"MEMORIES: {embeddings} n\n\ USER PROMPT: {prompt}",
        }
    )
    print(f"\n{len(embeddings)} message:response embeddings added for context.")


conversations = fetch_conversations()
create_vector_db(conversations=conversations)


while True:
    prompt = input(Fore.WHITE + "USER: \n")

    if prompt[:7].lower() == "/recall:":
        prompt = prompt[8:]
        recall(prompt=prompt)
        stream_response(prompt=prompt)

    elif prompt[:7] == "/forget:":
        remove_last_conversation()
        convo = convo[:-2]
        print("\n")

    elif prompt[:9].lower() == "/memorize:":
        prompt = prompt[10:]
        store_conversations(prompt=prompt, response="memory stored")
        print("\n")
    else:
        convo.append({"role": "user", "content": prompt})
        stream_response(prompt=prompt)

    stream_response(prompt=prompt)
