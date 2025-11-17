# advanced_features.py - Additional functionality
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import json

class AdvancedDocumentAnalyzer:
    """Extended functionality for document analysis."""
    
    def __init__(self, assistant):
        self.assistant = assistant
        self.summary_chain = self._create_summary_chain()
        self.keyword_chain = self._create_keyword_chain()
    
    def _create_summary_chain(self):
        """Create a chain for document summarization."""
        summary_prompt = PromptTemplate(
            input_variables=["text"],
            template="""
            Provide a comprehensive summary of the following text:
            
            {text}
            
            Summary should include:
            1. Main topics discussed
            2. Key findings or conclusions
            3. Important details
            
            Summary:
            """
        )
        return LLMChain(llm=self.assistant.llm, prompt=summary_prompt)
    
    def _create_keyword_chain(self):
        """Create a chain for keyword extraction."""
        keyword_prompt = PromptTemplate(
            input_variables=["text"],
            template="""
            Extract the most important keywords and phrases from this text.
            Return them as a JSON list.
            
            Text: {text}
            
            Keywords (JSON format):
            """
        )
        return LLMChain(llm=self.assistant.llm, prompt=keyword_prompt)
    
    def analyze_document_themes(self) -> Dict[str, Any]:
        """Analyze themes across all documents."""
        if not self.assistant.vectorstore:
            return {"error": "No documents processed"}
        
        # Get sample documents for analysis
        sample_docs = self.assistant.vectorstore.similarity_search("", k=5)
        combined_text = "\n\n".join([doc.page_content for doc in sample_docs])
        
        # Generate summary and keywords
        summary = self.summary_chain.run(combined_text[:4000])  # Limit for API
        keywords_raw = self.keyword_chain.run(combined_text[:2000])
        
        try:
            keywords = json.loads(keywords_raw)
        except:
            keywords = ["Analysis", "Documents", "Content"]
        
        return {
            "summary": summary,
            "keywords": keywords,
            "document_count": len(sample_docs)
        }