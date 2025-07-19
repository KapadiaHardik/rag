import os
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
import torch
import PyPDF2

class RAGSummarizer:
    def __init__(self, model_dir="models/"):
        # Load saved model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch.float16,
            device_map="cpu",
            trust_remote_code=True,
            attn_implementation="eager"
        )
        # Load sentence transformer for embeddings
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        # Initialize ChromaDB client
        self.client = chromadb.Client()
        self.collection_name = "document_chunks"
        # Create collection if it doesn't exist
        self.collection = self.client.create_collection(self.collection_name)
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=50,
            length_function=lambda x: len(self.tokenizer.encode(x))
        )

    def read_document(self, file_path):
        """Read content from a text or PDF file."""
        try:
            file_extension = os.path.splitext(file_path)[1].lower()
            if file_extension == '.txt':
                with open(file_path, 'r', encoding='utf-8') as file:
                    return file.read()
            elif file_extension == '.pdf':
                with open(file_path, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    text = ""
                    for page in reader.pages:
                        text += page.extract_text() or ""
                    return text
            else:
                raise ValueError("Unsupported file format. Use .txt or .pdf.")
        except Exception as e:
            raise ValueError(f"Error reading file {file_path}: {str(e)}")

    def chunk_text(self, text):
        """Split text into chunks using RecursiveCharacterTextSplitter."""
        chunks = self.text_splitter.split_text(text)
        return chunks if chunks else []

    def store_chunks(self, chunks):
        """Store chunks in ChromaDB with their embeddings."""
        # Try to delete the collection if it exists, ignore if it doesn't
        try:
            self.client.delete_collection(self.collection_name)
        except ValueError:
            pass
        # Recreate the collection
        self.collection = self.client.create_collection(self.collection_name)
        
        embeddings = self.embedder.encode(chunks, convert_to_tensor=False)
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            self.collection.add(
                documents=[chunk],
                embeddings=[embedding.tolist()],
                ids=[f"chunk_{i}"]
            )

    def retrieve_relant_chunks(self, query, top_k=3):
        """Retrieve top_k most relevant chunks from ChromaDB."""
        query_embedding = self.embedder.encode([query], convert_to_tensor=False)[0]
        # Adjust top_k to the number of available chunks
        chunk_count = self.collection.count()
        top_k = min(top_k, chunk_count) if chunk_count > 0 else 0
        
        if top_k == 0:
            return [], []
        
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k
        )
        chunks = results['documents'][0] if results['documents'] else []
        similarities = results['distances'][0] if results['distances'] else []
        return chunks, similarities

    def generate_summary(self, context, max_questions=3):
        """Generate summary using Phi-3.5 with an efficient prompt."""
        if not context:
            return "No relevant context available to summarize."
        
        prompt = f"""You are a professional document summarizer. Summarize the following document context into a concise summary of up to 200 words. Focus on key points, actionable insights, and main conclusions. Avoid unnecessary details and ensure clarity.

Context:
{context}

Summary:"""
        
        # Explicitly create attention mask
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
            padding=True,
            return_attention_mask=True
        )
        
        # Move inputs to CPU, keep input_ids as torch.long
        inputs = {
            "input_ids": inputs["input_ids"].to(device="cpu", dtype=torch.long),
            "attention_mask": inputs["attention_mask"].to(device="cpu", dtype=torch.long)
        }
        
        # Generate with minimal cache usage to avoid DynamicCache issues
        outputs = self.model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=250,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            num_return_sequences=1,
            use_cache=False  # Disable cache to bypass DynamicCache error
        )
        summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract only the summary part after the prompt
        summary = summary.split("Summary:")[1].strip() if "Summary:" in summary else summary
        return summary

    def summarize_document(self, file_path, query="Summarize the key points of this document"):
        """Main function to summarize a document from a file path using RAG."""
        # Read the document
        document = self.read_document(file_path)
        
        # Chunk the document
        chunks = self.chunk_text(document)
        if not chunks:
            return "Error: Document is empty or too short."
        
        # Store chunks in ChromaDB
        self.store_chunks(chunks)
        
        # Retrieve relevant chunks
        relevant_chunks, similarities = self.retrieve_relevant_chunks(query)
        
        # Combine relevant chunks into context
        context = "\n".join(relevant_chunks)
        
        # Generate summary
        summary = self.generate_summary(context)
        return summary

if __name__ == "__main__":
    # Example usage
    summarizer = RAGSummarizer()
    
    # Sample document path (replace with actual file path)
    sample_document_path = "sample_document.txt"
    
    # Create a sample document for testing if it doesn't exist
    if not os.path.exists(sample_document_path):
        with open(sample_document_path, 'w', encoding='utf-8') as f:
            f.write("""
            The financial report for Q3 2025 highlights a 15% revenue increase compared to Q2, driven by strong performance in the technology sector. The company expanded its market share in cloud computing, contributing to a 20% growth in service-based revenue. However, operational costs rose by 10% due to increased investment in R&D and supply chain disruptions. The board recommends optimizing supply chain processes to mitigate future risks. Net profit margins remained stable at 12%, supported by cost-cutting measures in non-core areas. The company plans to launch a new AI-driven product line in Q4, expected to boost revenue by 8%.
            """)
    
    summary = summarizer.summarize_document(sample_document_path)
    print("Generated Summary:")
    print(summary)