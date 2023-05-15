import os
import json
import tempfile
import boto3
import openai

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader

def lambda_handler(event, context):
    # S3クライアントの作成
    s3 = boto3.client('s3')
    
    # S3バケット名とオブジェクトキーを指定
    bucket_name = os.environ["BUCKET_NAME"]
    object_key = os.environ["OBJECT_KEY"]
    print("object_key: " + object_key)
    
    with tempfile.NamedTemporaryFile(suffix=".pdf") as temp_file:
        # S3からファイルをダウンロード
        s3.download_file(bucket_name, object_key, temp_file.name)
        
        # PyPDFLoaderを使用してPDFを展開
        loader = PyPDFLoader(temp_file.name)

        pages = loader.load_and_split()

    openai.api_key = os.environ["OPENAI_API_KEY"]
    index = []
    for page in pages:
        # ベクトル化を行う
        res = openai.Embedding.create(
            model='text-embedding-ada-002',
            input=page.page_content
        )

        # ベクトルをリストに追加
        index.append({
            'embedding': res['data'][0]['embedding']
        })

    print("Vectorize done!")
    # JSONに変換
    binary_vector_data = json.dumps(index)

    # バイナリデータをS3にアップロード
    s3.put_object(Body=binary_vector_data, Bucket=bucket_name, Key="faiss_vectors.json")
    print("DONE!")
