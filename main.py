import os
import sqlite3
import pandas as pd
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.chat_models import ChatOpenAI
import chromadb

class datachat():
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.db_path = './customer_churn.db'
        self.init_db()
        try:
          client = chromadb.Client()
          self.collection = client.get_or_create_collection(name="ddl_collection", metadata={"hnsw:space": "cosine"})
          print("ChromaDB Client initialized successfully.")
        except Exception as e:
          print(f"Error initializing ChromaDB Client: {e}")
          self.collection = None
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    def init_db(self):
        # Create the SQLite database from the dataset
        df = pd.read_excel(self.dataset_path)
        conn = sqlite3.connect(self.db_path)
        df.to_sql('customer_churn', conn, if_exists='replace', index=False)
        conn.close()

    def exe_sql(self, statement):
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query(statement, conn)
        conn.close()
        return df

    def get_meta(self, query):
        query_v = self.embeddings.embed_query(query)
        select_meta = self.collection.query(query_embeddings=query_v, n_results=8)
        return select_meta

    def extract_code(self, response):
        start = 0
        q = ""
        for line in response.splitlines():
            if '```sql' in line and start == 0:
                start = 1
            if '```' == line.strip() and start == 1:
                start = 0
                break
            if start == 1 and '```' not in line:
                q += '\n' + line
        return q.strip()

    def vectorize(self):
        statement = "SELECT sql FROM sqlite_master WHERE type = 'table';"
        df = self.exe_sql(statement)
        meta = df.to_dict(orient='records')

        documents_list = []
        embeddings_list = []
        ids_list = []

        for i, chunk in enumerate(meta):
            vector = self.embeddings.embed_query(str(chunk))
            documents_list.append(str(chunk))
            embeddings_list.append(vector)
            ids_list.append(f"ddl_{i}")

        self.collection.add(
            embeddings=embeddings_list,
            documents=documents_list,
            ids=ids_list
        )
        return

    def data_ops(self, query):
        # from langchain.chat_models import ChatOpenAI
        llm = ChatOpenAI(
            model_name="gpt-4o-mini",  # or "gpt-4"
            openai_api_key="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",
            temperature=0.0,
            max_tokens=4000
        )

        instruction = """
        As a SQL developer, generate a SQL statement for the query with reference to the database objects in the database schema.

        Database schema:
        {meta}

        Question: {input}

        Answer:
        """
        select_meta = self.get_meta(query)
        prompt_template = PromptTemplate.from_template(instruction)
        agent = LLMChain(llm=llm, prompt=prompt_template)
        response = agent.invoke({"input": query, "meta": select_meta})

        gen_sql = self.extract_code(response['text'])
        print(f"Generated SQL Query: {gen_sql}")
        df = self.exe_sql(gen_sql)
        return df


# Terminal-based interaction
if __name__ == "__main__":
    dataset_path = '/content/drive/MyDrive/Team Verizon_2/training and test data/mobile_subscriber_churn_train.xlsx'

    # Initialize the datachat object
    chatbot = datachat(dataset_path)

    # Vectorize the database
    print("Initializing the chatbot and vectorizing the database...")
    chatbot.vectorize()
    print("Database vectorized! You can now ask questions.")

    while True:
        # Take user input
        user_input = input("\nEnter your query (or type 'exit' to quit): ")
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break

        # Process the query
        try:
            response_df = chatbot.data_ops(user_input)
            print("\nQuery Results:")
            print(response_df)
        except Exception as e:
            print(f"An error occurred: {e}")