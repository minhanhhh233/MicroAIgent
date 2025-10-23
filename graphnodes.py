from langchain_nvidia_ai_endpoints import NVIDIARerank
import os
from multiagent import RetrieverManager
import io
from contextlib import redirect_stdout, redirect_stderr
from utils import automation
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv('API_KEY')

class Nodes:
    @staticmethod
    def retrieve(state):
        """
        Retrieve relevant documents from all services.
        """
        question = state["question"]
        print("---RETRIEVE---")
        # Get the already-initialized retriever (no ingestion!)
        retriever = RetrieverManager.get_retriever()        
        # Retrieve documents (fast vector search)
        documents = retriever.retrieve(question)
        
        # Print summary
        print(f"  Retrieved: {len(documents)} documents")
        
        # Group by service
        services = {}
        for doc in documents:
            service = doc.metadata.get('service', 'unknown')
            services[service] = services.get(service, 0) + 1
        
        print("  By service:")
        for service, count in services.items():
            print(f"    • {service}: {count} chunks")
        
        return {"documents": documents, "question": question}

    @staticmethod
    def rerank(state):
        print("NVIDIA--RERANKER")
        question = state["question"]
        documents = state["documents"]
        reranker =  NVIDIARerank(model="nvidia/llama-3.2-nv-rerankqa-1b-v2", api_key=api_key)
        documents = reranker.compress_documents(query=question, documents=documents)
        return {"documents": documents, "question": question}

    @staticmethod
    def generate(state):    
        print("GENERATE USING LLM")
        question = state["question"]
        documents = state["documents"]
        metrics_analysis = state["metrics_analysis"]

        # Format documents as context
        doc_context = "\n\n".join([
            f"Document {i+1}:\n{doc.page_content if hasattr(doc, 'page_content') else doc}"
            for i, doc in enumerate(documents)
        ])
        
        # Add metrics analysis to context
        if metrics_analysis:
            combined_context = f"""{doc_context}
            
            --- METRICS ANALYSIS ---
            {metrics_analysis}
"""
        else:
            combined_context = doc_context

        generation = automation.rag_chain.invoke({"context": combined_context, "question": question})
        return {"documents": documents, "question": question, "generation": generation}

    @staticmethod
    def grade_documents(state):    
        print("CHECKING DOCUMENT RELEVANCE TO QUESTION")
        question = state["question"]
        ret_documents = state["documents"]

        filtered_docs = []
        for doc in ret_documents:
            score = automation.retrieval_grader.invoke(
                {"question": question, "document": doc.page_content}
            )
            grade = score.binary_score
            if grade == "yes":
                print("---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(doc)
            else:
                print("---GRADE: DOCUMENT NOT RELEVANT---")
        return {"documents": filtered_docs, "question": question}

    @staticmethod
    def transform_query(state):
        
        print("REWRITE PROMPT")
        question = state["question"]
        documents = state["documents"]

        better_question = automation.question_rewriter.invoke({"question": question})
        print(f"actual query : {question} \n Transformed query:{better_question}")
        return {"documents": documents, "question": better_question}
