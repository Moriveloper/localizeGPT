import os
import json
import tempfile
import boto3
import openai
import langchain
import requests
import pdfplumber

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from io import BytesIO
from llama_index import GPTSimpleVectorIndex, SimpleDirectoryReader

url = "https://www.mcf.bz/2021/12/31/post-4602/"

summarize_prompt_template = """以下の文章を簡潔に要約してください。:

{text}

要約:"""

qa_prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Answer in Japanese:"""

# Webページからテキストを抽出
def get_webpage_texts(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    text_list = []
    for string in soup.stripped_strings:
        text_list.append(string)
    return text_list

# PDFページからテキストを抽出
def get_pdf_text(url):
    response = requests.get(url)
    pdf_file = BytesIO(response.content)
    text_list = []
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text_list.append(page.extract_text())
    return text_list

# URLをフォルダ名に変更
def modify_url_to_s3_path(url):
    modified_string = url.replace("https://", "")
    modified_string = modified_string.replace("/", "-")
    modified_string = modified_string.rstrip("-")
    modified_string = modified_string.rstrip(".")
    return modified_string

# S3にディレクトリを格納
def upload_dir_s3(dirpath, s3bucket, s3_path):
    for root,dirs,files in os.walk(dirpath):
        for file in files:
            s3_key = os.path.join(root, file).replace(dirpath, s3_path, 1)
            s3bucket.upload_file(os.path.join(root, file), s3_key)

# S3からディレクトリを取得
def download_dir_s3(dirpath, s3bucket):
    for obj in s3bucket.objects.filter(Prefix = dirpath):
        if not os.path.exists(os.path.join('/tmp/', os.path.dirname(obj.key))):
            os.makedirs(os.path.join('/tmp/', os.path.dirname(obj.key)))
        s3bucket.download_file(obj.key, os.path.join('/tmp/', obj.key))
    return os.path.join('/tmp/', dirpath)

# DynamoDBへ保存処理
def write_to_dynamo(url, s3_path):
    dynamodb = boto3.resource('dynamodb')
    table_name = os.environ["TABLE_NAME"]
    table = dynamodb.Table(table_name)
    now = datetime.now()

    # データを書き込む
    table.put_item(
        Item={
            'Url': url,
            'AttributeType': "S3Key",
            'AttributeValue': s3_path,
            'created_at': now.isoformat()
        }
    )

# DynamoDBからの取得処理
def get_from_dynamo(url):
    dynamodb = boto3.resource('dynamodb')
    table_name = os.environ["TABLE_NAME"]
    table = dynamodb.Table(table_name)

    # データを取得する
    response = table.get_item(
        Key={
            'Url': url,
            'AttributeType': "S3Key"
        }
    )

    # 取得した項目が存在するかを確認
    if 'Item' in response:
        return response['Item']
    else:
        return None

def lambda_handler(event, context):
    # S3の準備
    s3 = boto3.resource('s3')
    bucket_name = os.environ["BUCKET_NAME"]
    s3_bucket = s3.Bucket(bucket_name)

    # ChatGPTの準備
    openai.api_key = os.environ["OPENAI_API_KEY"]
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    embeddings = OpenAIEmbeddings()

    url_contain = False

    if url_contain:
        # 投稿にurlが含まれているので要約を行う
        print("Summarize.")

        # TODO 要約およびデータ学習中ですと返信する

        # テキストを取得してtext_listに格納
        text_list = []
        if urlparse(url).path.endswith('.pdf'):
            text_list.extend(get_pdf_text(url))
        else:
            text_list.extend(get_webpage_texts(url))
        print("text_list: " + text_list[1])

        # テキストを学習用に分割する
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_text(text_list)
        docs = [Document(page_content=t) for t in texts[:3]]

        要約処理
        summarize_template = PromptTemplate(
                    template=summarize_prompt_template, input_variables=["text"])
        chain = load_summarize_chain(llm, chain_type="map_reduce", map_prompt=summarize_template, combine_prompt=summarize_template)
        summarize_result = chain.run(docs)
        print("summarize_result: " + summarize_result)

        ベクトルデータ化してindexを作成する
        dbCreate = FAISS.from_documents(docs, embeddings)
        dbCreate.save_local("/tmp/tmp_index")
        print("Vectorize done!")

        # S3へアップロードする
        s3_upload_path = modify_url_to_s3_path(url)
        upload_dir_s3("/tmp/tmp_index", s3_bucket, s3_upload_path)
        print("S3 uploaded: " + s3_upload_path)

        # DynamoDBへアップロードする
        write_to_dynamo(url, s3_path)
        print("DynamoDB write done!")

        # TODO summarize_resultを返信する
    else:
        # 投稿自体にurlが含まれていないので質問と判断
        print("QA.")

        # DynamoDBから該当urlのベクトルデータが保存されているS3の情報を取得する
        item = get_from_dynamo(url)
        if item is not None:
            print("DynamoDB item: " + json.dumps(item))
            s3_download_path = item["AttributeValue"]
            print(s3_download_path)

            # S3から保存済みのVectorIndexを取得
            index_dir = download_dir_s3(s3_download_path, s3_bucket)
            dbDownload = FAISS.load_local(index_dir, embeddings)

            # QA処理
            qa_prompt = PromptTemplate(
                template=qa_prompt_template, input_variables=["context", "question"]
            )
            chain_type_kwargs = {"prompt": qa_prompt}
            qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=dbDownload.as_retriever(), chain_type_kwargs=chain_type_kwargs)
            query = "MCFの強さとは？"
            qa_result = qa.run(query)
            print(qa_result)
        else:
            print("DynamoDB item not found.")
            # TODO どのURLに対する質問かわからなかったよと返信する


