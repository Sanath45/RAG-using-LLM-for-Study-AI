from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import re
import os

class PDFProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # Increased chunk size
            chunk_overlap=200,  # Increased overlap
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", " ", ""],
            is_separator_regex=False
        )
    
    def clean_text(self, text):
        # Preserve marketing-specific elements like bullets, numbers, and special formatting
        # Remove multiple spaces but preserve paragraph structure
        text = re.sub(r' +', ' ', text)
        text = re.sub(r'\n+', '\n', text)
        
        # Preserve bullets and numbering
        text = re.sub(r'([•\-*]) ', r'\n\1 ', text)
        text = re.sub(r'(\d+\.) ', r'\n\1 ', text)
        
        # Preserve quotes that may be testimonials
        text = re.sub(r'["""]([^"""]+)["""]', r'" \1 "', text)
        
        # Preserve statistical information (numbers, percentages)
        text = re.sub(r'(\d+(\.\d+)?%)', r' \1 ', text)
        
        # Remove special characters but keep marketing-relevant punctuation
        text = re.sub(r'[^\w\s.,!?;:\-–—"\'()%$#@&*]', ' ', text)
        
        return text.strip()
    
    def process_pdf(self, file_path):
        try:
            print(f"Starting to process PDF: {file_path}")
            if not os.path.exists(file_path):
                print(f"File not found: {file_path}")
                return [], []
            
            # Attempt to open and read the PDF
            pdf_reader = PdfReader(file_path)
            print(f"PDF loaded with {len(pdf_reader.pages)} pages")
            full_text = ""
            
            # Extract text page by page with metadata
            pages_text = []
            for page_num, page in enumerate(pdf_reader.pages):
                text = page.extract_text()
                if text:
                    cleaned_text = self.clean_text(text)
                    print(f"Page {page_num + 1}: Extracted {len(cleaned_text)} characters")
                    pages_text.append({
                        'content': cleaned_text,
                        'page': page_num + 1
                    })
                    full_text += cleaned_text + "\n\n"
                else:
                    print(f"Page {page_num + 1}: No text extracted")
            
            if not full_text.strip():
                print("No text content extracted from PDF")
                return [], []
            
            print(f"Total extracted text: {len(full_text)} characters")
            
            # Ensure we're creating chunks with actual content
            if len(full_text) < 100:
                # If text is very short, create a single document without splitting
                print("Text too short for splitting, creating single document")
                document = Document(
                    page_content=full_text,
                    metadata={"source": file_path, "page_number": 1, "chunk_id": 0}
                )
                return [full_text], [document]
            
            # Create documents with meaningful chunks
            print("Creating text chunks...")
            chunks = self.text_splitter.create_documents(
                texts=[full_text],
                metadatas=[{"source": file_path}]
            )
            print(f"Created {len(chunks)} chunks from text")
            
            # Add page numbers and section info to chunks
            documents = []
            for i, chunk in enumerate(chunks):
                # Find the page number for this chunk
                page_num = self.find_page_number(chunk.page_content, pages_text) or 1
                # Try to identify content type (heading, paragraph, list, etc.)
                content_type = self.identify_content_type(chunk.page_content)
                chunk.metadata.update({
                    "chunk_id": i,
                    "page_number": page_num,
                    "content_type": content_type
                })
                documents.append(chunk)
                print(f"Chunk {i}: {len(chunk.page_content)} chars, Page {page_num}, Type {content_type}")
            
            # Ensure we have actual chunks
            if not documents:
                print("No documents created from chunks")
                return [], []
            
            return [doc.page_content for doc in documents], documents
            
        except Exception as e:
            print(f"Error processing PDF: {str(e)}")
            import traceback
            traceback.print_exc()
            return [], []
    
    def find_page_number(self, chunk_text, pages_text):
        # Find which page contains this chunk
        for page_info in pages_text:
            if chunk_text in page_info['content']:
                return page_info['page']
        # If not found by direct matching (which can fail due to whitespace differences),
        # try a more lenient approach
        for page_info in pages_text:
            # Check if a significant portion of the chunk is in this page
            # This helps with chunks that span page boundaries
            cleaned_chunk = re.sub(r'\s+', '', chunk_text)
            cleaned_page = re.sub(r'\s+', '', page_info['content'])
            
            # Look for 80% or more of the chunk text in this page
            if len(cleaned_chunk) > 10:  # Only check meaningful chunks
                matches = 0
                # Count character matches (simplified approach)
                for i in range(0, len(cleaned_chunk), 10):
                    segment = cleaned_chunk[i:i+10]
                    if segment in cleaned_page:
                        matches += len(segment)
                
                if matches >= len(cleaned_chunk) * 0.6:  # 60% threshold
                    return page_info['page']
        
        return 1  # Default to page 1 if not found
    
    def identify_content_type(self, text):
        """Identify the type of content in the chunk for better context"""
        # Check if it's likely a heading
        if len(text) < 100 and (text.endswith(':') or text.isupper()):
            return "heading"
        # Check if it's a list (contains bullet points or numbered items)
        elif re.search(r'(^|\n)[\s]*[•\-*][\s]+\w+', text) or re.search(r'(^|\n)[\s]*\d+\.[\s]+\w+', text):
            return "list"
        # Check if it contains statistical data
        elif re.search(r'\d+(\.\d+)?%', text) or re.search(r'\$\d+', text):
            return "statistics"
        # Check if it's likely a testimonial or quote
        elif re.search(r'["""]([^"""]{10,})["""]', text):
            return "quote"
        # Default to paragraph
        else:
            return "paragraph"