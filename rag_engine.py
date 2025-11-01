from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import BaseMessage
from typing import List, Dict
import os
import json
import re
import random

class RAGEngine:
    def __init__(self, vector_store_path='vector_store', processed_files_log='processed_files.json'):
        # Initialize Azure OpenAI settings
        if not all([
            os.getenv('AZURE_OPENAI_API_KEY'),
            os.getenv('AZURE_OPENAI_ENDPOINT'),
            os.getenv('AZURE_DEPLOYMENT_NAME'),
            os.getenv('AZURE_EMBEDDINGS_DEPLOYMENT_NAME')
        ]):
            raise ValueError("Azure OpenAI configuration not found in environment variables")
            
        # Initialize Azure OpenAI embeddings
        self.embeddings = AzureOpenAIEmbeddings(
            azure_deployment=os.getenv('AZURE_EMBEDDINGS_DEPLOYMENT_NAME'),
            openai_api_version="2023-05-15"
        )
        
        self.chat_history = []
        self.vector_store = None
        self.qa_chain = None
        self.loaded_files = {}  # Track loaded PDFs
        self.content_mode = "general"  # Default to general mode
        
        # Base academic prompt template
        self.ACADEMIC_BASE_PROMPT = PromptTemplate(
            template="""You are an expert academic assistant for master's degree students. 
            Your task is to provide detailed, accurate information and guidance based on the available information.

            {context}

            Chat History:
            {chat_history}

            Current Request: {question}

            If the request is about specific document content and relevant information is available in the context, base your response primarily on that information.
            If no relevant information is found or no documents are uploaded, use your general knowledge to provide a helpful, accurate response.

            Your response should be:
            1. Well-structured and academically rigorous
            2. Properly cited when referencing specific materials
            3. Include relevant data, formulas, or methodology when appropriate
            4. Follow academic conventions and best practices
            5. Formatted with appropriate sections and clarity

            Response:""",
            input_variables=["context", "chat_history", "question"]
        )
        
        # Content-specific templates
        self.CONTENT_TEMPLATES = {
            "math": PromptTemplate(
                template="""You are tasked with solving a mathematical problem step-by-step.

                {context}

                Chat History:
                {chat_history}

                Current Request: {question}

                Follow these guidelines:
                1. Identify the type of mathematical problem
                2. Break down the problem into clear, logical steps
                3. Explain each step thoroughly with the relevant formulas and properties
                4. Show all calculations explicitly
                5. Verify the answer when possible
                6. Highlight key concepts being applied
                7. If multiple approaches are possible, explain the most efficient one
                8. Use proper mathematical notation (use markdown formatting for equations)

                Response:""",
                input_variables=["context", "chat_history", "question"]
            ),
            "literature": PromptTemplate(
                template="""You are providing a literature analysis or explanation.

                {context}

                Chat History:
                {chat_history}

                Current Request: {question}

                Follow these guidelines:
                1. Analyze the text with depth and critical insight
                2. Identify key themes, literary devices, and authorial intent
                3. Place the work in its proper historical and cultural context
                4. Reference relevant literary theories or frameworks when appropriate
                5. Include textual evidence to support claims
                6. Consider multiple interpretations when relevant
                7. Use academic literary terminology correctly
                8. Structure the analysis logically with clear arguments

                Response:""",
                input_variables=["context", "chat_history", "question"]
            ),
            "science": PromptTemplate(
                template="""You are explaining a scientific concept or analyzing scientific data.

                {context}

                Chat History:
                {chat_history}

                Current Request: {question}

                Follow these guidelines:
                1. Explain scientific concepts with precision and accuracy
                2. Break down complex processes into understandable components
                3. Include relevant scientific principles, laws, or theories
                4. Reference experimental evidence when available
                5. Distinguish between established facts and theoretical models
                6. Use proper scientific terminology and units
                7. Include visual representations or diagrams when helpful (describe them textually)
                8. Address limitations or uncertainties in current scientific understanding

                Response:""",
                input_variables=["context", "chat_history", "question"]
            ),
            "research": PromptTemplate(
                template="""You are providing guidance on academic research methodology.

                {context}

                Chat History:
                {chat_history}

                Current Request: {question}

                Follow these guidelines:
                1. Explain research methodologies clearly and accurately
                2. Provide guidance on research design appropriate to the field
                3. Discuss data collection and analysis techniques
                4. Address research ethics considerations
                5. Suggest appropriate statistical approaches when relevant
                6. Recommend best practices for academic writing and citation
                7. Discuss strategies for literature review
                8. Include practical advice for implementation

                Response:""",
                input_variables=["context", "chat_history", "question"]
            ),
            "summary": PromptTemplate(
                template="""You are creating a concise academic summary.

                {context}

                Chat History:
                {chat_history}

                Current Request: {question}

                Follow these guidelines:
                1. Identify and include only the most essential information
                2. Maintain academic tone and terminology
                3. Structure logically with clear introduction, body, and conclusion
                4. Preserve the key arguments and evidence from the original
                5. Avoid adding personal opinions or interpretations
                6. Keep focus on the main thesis and supporting points
                7. Use transition phrases to maintain coherence
                8. Ensure accuracy in representing the original source

                Response:""",
                input_variables=["context", "chat_history", "question"]
            )
        }

        # MCQ test generation template
        self.MCQ_TEMPLATE = PromptTemplate(
            template="""Generate {num_questions} multiple-choice questions about {subject} at a master's degree level.

            For each question:
            1. Provide a clear, challenging question
            2. Include 4 possible answers labeled A, B, C, and D
            3. Indicate the correct answer
            4. Include a brief explanation of why the answer is correct
            5. Ensure questions test different cognitive levels (knowledge, comprehension, application, analysis)
            6. Focus on important concepts in {subject}

            Format as a JSON array with each question as an object containing fields:
            - question: the question text
            - options: array of 4 options
            - correct_answer: the letter (A, B, C, or D) of the correct option
            - explanation: why the answer is correct

            The output should ONLY contain valid JSON with no additional text.

            Response:""",
            input_variables=["num_questions", "subject"]
        )

        # Academic greeting patterns
        self.greeting_patterns = {
            'hi': 'Hello! I\'m your academic assistant. I can help with understanding complex concepts, solving problems, analyzing texts, and providing study guidance. What would you like to learn about today?',
            'hello': 'Hi there! I\'m ready to assist with your academic needs - whether it\'s explaining theories, solving equations, analyzing literature, or providing research guidance. How can I help you?',
            'help': '''I can help you with various academic tasks:

            Learning Support:
            • Explaining complex concepts and theories
            • Solving mathematical and scientific problems step-by-step
            • Analyzing literature and research papers
            • Summarizing academic content
            • Providing study strategies and exam preparation

            Research Assistance:
            • Research methodology guidance
            • Literature review suggestions
            • Data analysis explanations
            • Academic writing tips
            • Citation and reference help

            Just describe what you're studying or what problem you need help with!'''
        }

        self.vector_store_path = vector_store_path
        self.processed_files_log = processed_files_log
        
        # Load existing vector store if it exists
        index_path = os.path.join(vector_store_path, 'index.faiss')
        if os.path.exists(index_path):
            try:
                print(f"Loading vector store from {vector_store_path}")
                self.vector_store = FAISS.load_local(
                    vector_store_path, 
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                print("Vector store loaded successfully")
                self._initialize_qa_chain()
            except Exception as e:
                print(f"Error loading vector store: {e}")
                self.vector_store = None
        else:
            print(f"No existing vector store found at {index_path}")
            self.vector_store = None
        
        # Load processed files log
        self.loaded_files = self._load_processed_files()

        # Initialize conversation memory with a window of 10 exchanges (20 messages)
        self.memory = ConversationBufferWindowMemory(
            k=10,  # Keep last 10 exchanges
            memory_key="chat_history",
            output_key="answer",
            return_messages=True
        )
        
        # Session-based memory storage for multiple conversations
        self.session_memories = {}

    def _load_processed_files(self):
        """Load the list of processed files from disk"""
        try:
            if os.path.exists(self.processed_files_log):
                with open(self.processed_files_log, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            print(f"Error loading processed files log: {e}")
            return {}

    def _save_processed_files(self):
        """Save the list of processed files to disk"""
        try:
            with open(self.processed_files_log, 'w') as f:
                json.dump(self.loaded_files, f)
        except Exception as e:
            print(f"Error saving processed files log: {e}")

    def is_file_processed(self, filename: str) -> bool:
        """Check if a file has already been processed"""
        return filename in self.loaded_files

    def create_index(self, texts: List[str], documents=None):
        try:
            # Create FAISS vector store with metadata
            self.vector_store = FAISS.from_texts(
                texts,
                self.embeddings,
                metadatas=[doc.metadata for doc in documents] if documents else None
            )
            
            # Initialize QA chain with Azure OpenAI
            self._initialize_qa_chain()
            return True
        except Exception as e:
            print(f"Error creating index: {str(e)}")
            return False

    def add_to_index(self, texts: List[str], documents=None, file_name: str = None):
        try:
            print(f"Adding to index: file_name={file_name}, texts={len(texts)}, documents={len(documents) if documents else 0}")
            
            # Validate inputs
            if not texts or not documents:
                print("No texts or documents provided for indexing")
                return False
                
            # Skip if file already processed
            if self.is_file_processed(file_name):
                print(f"File '{file_name}' already processed, skipping")
                return True
            
            # Debug documents content
            for i, doc in enumerate(documents[:3]):  # Print first 3 docs for debugging
                print(f"Document {i} - Length: {len(doc.page_content)}, Metadata: {doc.metadata}")
            
            # Check for potential embedding issues
            for i, text in enumerate(texts):
                if not text.strip():
                    print(f"Warning: Empty text at index {i}")
                if len(text) < 10:
                    print(f"Warning: Very short text at index {i}: '{text}'")
            
            # Create new embeddings for the current document
            print("Creating vectors for new documents...")
            try:
                new_vectors = FAISS.from_texts(
                    texts,
                    self.embeddings,
                    metadatas=[{**doc.metadata, "file_name": file_name} for doc in documents] if documents else None
                )
                print(f"Successfully created vectors for {len(texts)} texts")
            except Exception as e:
                print(f"Error creating vectors: {str(e)}")
                import traceback
                traceback.print_exc()
                return False
            
            # Merge with existing index if it exists
            try:
                if self.vector_store is None:
                    print("No existing vector store, using new vectors directly")
                    self.vector_store = new_vectors
                else:
                    print(f"Merging new vectors with existing index ({len(texts)} new documents)")
                    self.vector_store.merge_from(new_vectors)
                
                # Save vector store
                if self.vector_store_path:
                    print(f"Saving vector store to {self.vector_store_path}")
                    os.makedirs(self.vector_store_path, exist_ok=True)  # Create directory if it doesn't exist
                    self.vector_store.save_local(self.vector_store_path)
                    print("Vector store saved successfully")
                
                # Track loaded file
                if file_name:
                    self.loaded_files[file_name] = len(texts)
                    self._save_processed_files()
                    print(f"Updated processed files log: {file_name} with {len(texts)} chunks")
                
                # Reinitialize QA chain with updated vector store
                print("Initializing QA chain with updated vector store")
                self._initialize_qa_chain()
                return True
            except Exception as e:
                print(f"Error updating or saving vector store: {str(e)}")
                import traceback
                traceback.print_exc()
                return False
        except Exception as e:
            print(f"Error in add_to_index: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def remove_file(self, filename: str) -> bool:
        """Remove a file from the index and loaded files tracking"""
        try:
            if not self.is_file_processed(filename):
                print(f"File {filename} not found in processed files")
                return False

            # Remove from loaded files tracking
            if filename in self.loaded_files:
                del self.loaded_files[filename]
                self._save_processed_files()
                print(f"Removed {filename} from loaded files tracking")

            if self.vector_store is None:
                print("No vector store exists")
                return True

            # Create new vector store without the removed file's vectors
            new_vectors = None
            doc_counter = 0

            for doc in self.vector_store.docstore._dict.values():
                if doc.metadata.get('file_name') != filename:
                    doc_counter += 1
                    try:
                        if new_vectors is None:
                            new_vectors = FAISS.from_texts(
                                [doc.page_content],
                                self.embeddings,
                                metadatas=[doc.metadata]
                            )
                        else:
                            temp_vectors = FAISS.from_texts(
                                [doc.page_content],
                                self.embeddings,
                                metadatas=[doc.metadata]
                            )
                            new_vectors.merge_from(temp_vectors)
                    except Exception as e:
                        print(f"Error processing document during removal: {e}")
                        continue

            # Update vector store
            if doc_counter > 0 and new_vectors is not None:
                self.vector_store = new_vectors
                # Save updated vector store
                if self.vector_store_path:
                    os.makedirs(self.vector_store_path, exist_ok=True)
                    self.vector_store.save_local(self.vector_store_path)
                self._initialize_qa_chain()
                print(f"Vector store updated, removed {filename}")
            else:
                # If no documents left, reset vector store
                self.vector_store = None
                self.qa_chain = None
                print("Vector store reset - no documents remaining")

            return True

        except Exception as e:
            print(f"Error removing file: {e}")
            import traceback
            traceback.print_exc()
            return False

    def get_or_create_session_memory(self, session_id: str = "default") -> ConversationBufferWindowMemory:
        """Get or create memory for a specific session"""
        if session_id not in self.session_memories:
            self.session_memories[session_id] = ConversationBufferWindowMemory(
                k=10,  # Keep last 10 exchanges
                memory_key="chat_history",
                output_key="answer", 
                return_messages=True
            )
        return self.session_memories[session_id]

    def clear_session_memory(self, session_id: str = "default"):
        """Clear memory for a specific session"""
        if session_id in self.session_memories:
            self.session_memories[session_id].clear()
        else:
            # Create new memory if it doesn't exist
            self.session_memories[session_id] = ConversationBufferWindowMemory(
                k=10,
                memory_key="chat_history", 
                output_key="answer",
                return_messages=True
            )

    def get_session_history(self, session_id: str = "default") -> List[BaseMessage]:
        """Get conversation history for a session"""
        memory = self.get_or_create_session_memory(session_id)
        return memory.chat_memory.messages

    def add_to_session_memory(self, session_id: str, human_message: str, ai_message: str):
        """Manually add messages to session memory"""
        memory = self.get_or_create_session_memory(session_id)
        memory.chat_memory.add_user_message(human_message)
        memory.chat_memory.add_ai_message(ai_message)

    def _initialize_qa_chain(self):
        """Initialize or reinitialize the QA chain with current vector store"""
        if not self.vector_store:
            self.qa_chain = None
            return
            
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=AzureChatOpenAI(
                temperature=0.3,  # Lower temperature for academic accuracy
                deployment_name=os.getenv('AZURE_DEPLOYMENT_NAME'),
                model_name="gpt-35-turbo-16k",
                openai_api_version="2023-05-15"
            ),
            retriever=self.vector_store.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": 5,
                    "fetch_k": 8,
                    "lambda_mult": 0.7
                }
            ),
            combine_docs_chain_kwargs={"prompt": self.ACADEMIC_BASE_PROMPT},
            return_source_documents=False,
            verbose=True,
            memory=self.memory  # Use the default memory for backward compatibility
        )

    def get_response(self, query: str, content_params=None, session_id: str = "default"):
        try:
            # Check for greetings first
            query_lower = query.lower().strip()
            if query_lower in self.greeting_patterns:
                return self.greeting_patterns[query_lower], None
            
            # Process content parameters
            if not content_params:
                content_params = {}
                
            # Detect content type if not specified
            content_type = content_params.get("content_type") or self.detect_content_type(query)
            
            # Get or create session memory
            session_memory = self.get_or_create_session_memory(session_id)
            
            # If no QA chain exists (no documents loaded) or explicitly requesting general mode
            if not self.qa_chain or content_params.get("mode") == "general":
                # Use general mode with Azure OpenAI directly
                prompt_template = self.CONTENT_TEMPLATES.get(content_type, self.ACADEMIC_BASE_PROMPT)
                
                llm = AzureChatOpenAI(
                    temperature=0.3,  # Lower temperature for academic accuracy
                    deployment_name=os.getenv('AZURE_DEPLOYMENT_NAME'),
                    model_name="gpt-35-turbo-16k",
                    openai_api_version="2023-05-15"
                )
                
                # Format chat history for the prompt
                chat_history_text = ""
                if session_memory.chat_memory.messages:
                    for message in session_memory.chat_memory.messages[-10:]:  # Last 10 messages
                        if isinstance(message, HumanMessage):
                            chat_history_text += f"Human: {message.content}\n"
                        elif isinstance(message, AIMessage):
                            chat_history_text += f"Assistant: {message.content}\n"
                
                # Create the prompt with context and history
                formatted_prompt = prompt_template.template.replace(
                    '{context}', 'Use your general academic knowledge to answer.'
                ).replace(
                    '{chat_history}', chat_history_text
                ).replace(
                    '{question}', query
                )
                
                # Get response directly from LLM
                response = llm.invoke(formatted_prompt)
                answer = response.content
                
                # Save to session memory
                session_memory.chat_memory.add_user_message(query)
                session_memory.chat_memory.add_ai_message(answer)
                
                return answer, None
            
            # Use RAG with loaded documents
            prompt_template = self.CONTENT_TEMPLATES.get(content_type, self.ACADEMIC_BASE_PROMPT)
            
            # Create a temporary QA chain with the appropriate prompt and session memory
            temp_qa_chain = ConversationalRetrievalChain.from_llm(
                llm=AzureChatOpenAI(
                    temperature=0.3,  # Lower temperature for academic accuracy
                    deployment_name=os.getenv('AZURE_DEPLOYMENT_NAME'),
                    model_name="gpt-35-turbo-16k",
                    openai_api_version="2023-05-15"
                ),
                retriever=self.vector_store.as_retriever(
                    search_type="mmr",
                    search_kwargs={"k": 5, "fetch_k": 8, "lambda_mult": 0.7}
                ),
                combine_docs_chain_kwargs={"prompt": prompt_template},
                return_source_documents=False,
                memory=session_memory  # Use session-specific memory
            )
            
            # Get response using RAG with memory
            result = temp_qa_chain.invoke({
                "question": query
            })
            
            answer = result['answer']
            return answer, None
            
        except Exception as e:
            print(f"Error getting response: {str(e)}")
            import traceback
            traceback.print_exc()
            return "I'm here to help with your academic questions! Please try asking a specific question.", None

    def detect_content_type(self, query: str) -> str:
        """Detect the type of academic content being requested"""
        query_lower = query.lower()
        
        # Math detection
        math_keywords = ["solve", "equation", "calculate", "formula", "derivative", "integral", "mathematics", 
                        "theorem", "function", "graph", "algebra", "calculus", "geometry", "statistics"]
        if any(keyword in query_lower for keyword in math_keywords):
            return "math"
            
        # Literature detection
        literature_keywords = ["analyze", "interpret", "theme", "character", "literary", "novel", "poem", 
                              "symbolism", "author", "text", "narrative", "fiction", "literature"]
        if any(keyword in query_lower for keyword in literature_keywords):
            return "literature"
            
        # Science detection
        science_keywords = ["experiment", "theory", "hypothesis", "observation", "scientific", "physics", 
                           "chemistry", "biology", "molecule", "cell", "organism", "energy", "reaction"]
        if any(keyword in query_lower for keyword in science_keywords):
            return "science"
            
        # Research detection
        research_keywords = ["methodology", "research", "study", "analysis", "data", "survey", "interview", 
                            "sample", "population", "variable", "validity", "reliability", "qualitative", "quantitative"]
        if any(keyword in query_lower for keyword in research_keywords):
            return "research"
            
        # Summary detection
        summary_keywords = ["summarize", "summary", "overview", "brief", "key points", "main ideas", 
                           "synopsis", "recap", "condense", "digest"]
        if any(keyword in query_lower for keyword in summary_keywords):
            return "summary"
            
        # Default to general academic
        return "general"

    def generate_mcq_test(self, subject: str, num_questions: int = 5):
        """Generate an MCQ test on a specific subject"""
        try:
            llm = AzureChatOpenAI(
                temperature=0.5,  # Slightly higher for diverse questions
                deployment_name=os.getenv('AZURE_DEPLOYMENT_NAME'),
                model_name="gpt-35-turbo-16k",
                openai_api_version="2023-05-15"
            )
            
            # Format prompt for MCQ generation
            formatted_prompt = self.MCQ_TEMPLATE.format(
                num_questions=min(num_questions, 20),  # Limit to max 20 questions
                subject=subject
            )
            
            # Get response directly from LLM
            response = llm.invoke(formatted_prompt)
            mcq_content = response.content
            
            # Process the JSON response
            try:
                # Extract JSON from the response (in case there's text before/after)
                json_match = re.search(r'(\[.*\])', mcq_content, re.DOTALL)
                if json_match:
                    mcq_content = json_match.group(1)
                
                mcq_data = json.loads(mcq_content)
                
                # Make sure it's properly formatted
                formatted_mcq = []
                for i, question in enumerate(mcq_data):
                    formatted_question = {
                        'id': i + 1,
                        'question': question.get('question', ''),
                        'options': question.get('options', []),
                        'correct_answer': question.get('correct_answer', ''),
                        'explanation': question.get('explanation', '')
                    }
                    formatted_mcq.append(formatted_question)
                
                return formatted_mcq
                
            except json.JSONDecodeError as e:
                print(f"Error parsing MCQ JSON: {str(e)}")
                print(f"Raw content: {mcq_content}")
                # Try to manually extract questions as fallback
                return self._extract_questions_fallback(mcq_content)
                
        except Exception as e:
            print(f"Error generating MCQ test: {str(e)}")
            import traceback
            traceback.print_exc()
            return []
    
    def _extract_questions_fallback(self, content):
        """Fallback method to extract questions if JSON parsing fails"""
        questions = []
        current_question = None
        question_pattern = re.compile(r'(\d+)\.\s+(.*?)$')
        option_pattern = re.compile(r'([A-D])\)\s+(.*?)$')
        correct_pattern = re.compile(r'Correct Answer:\s*([A-D])')
        explanation_pattern = re.compile(r'Explanation:\s*(.*?)$')
        
        for line in content.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            # Check if this is a new question
            question_match = question_pattern.match(line)
            if question_match:
                # Save previous question if it exists
                if current_question:
                    questions.append(current_question)
                
                # Start new question
                q_num = int(question_match.group(1))
                q_text = question_match.group(2)
                current_question = {
                    'id': q_num,
                    'question': q_text,
                    'options': [],
                    'correct_answer': '',
                    'explanation': ''
                }
                continue
                
            # Check if this is an option
            option_match = option_pattern.match(line)
            if option_match and current_question:
                option_letter = option_match.group(1)
                option_text = option_match.group(2)
                current_question['options'].append(f"{option_letter}) {option_text}")
                continue
                
            # Check if this is the correct answer
            correct_match = correct_pattern.match(line)
            if correct_match and current_question:
                current_question['correct_answer'] = correct_match.group(1)
                continue
                
            # Check if this is the explanation
            explanation_match = explanation_pattern.match(line)
            if explanation_match and current_question:
                current_question['explanation'] = explanation_match.group(1)
                continue
                
            # If line contains "explanation" case-insensitive
            if "explanation" in line.lower() and current_question:
                parts = line.split(":", 1)
                if len(parts) > 1:
                    current_question['explanation'] = parts[1].strip()
        
        # Add the last question
        if current_question:
            questions.append(current_question)
            
        return questions

    def evaluate_mcq_answers(self, mcq_test, user_answers):
        """Evaluate user answers for an MCQ test"""
        if not mcq_test or not user_answers:
            return {"score": 0, "total": 0, "details": []}
            
        results = {
            "score": 0,
            "total": len(mcq_test),
            "details": []
        }
        
        for question in mcq_test:
            q_id = question['id']
            user_answer = user_answers.get(str(q_id), "")
            correct = question['correct_answer'] == user_answer
            
            if correct:
                results["score"] += 1
                
            results["details"].append({
                "question_id": q_id,
                "correct": correct,
                "user_answer": user_answer,
                "correct_answer": question['correct_answer'],
                "explanation": question['explanation']
            })
            
        # Calculate percentage
        results["percentage"] = (results["score"] / results["total"]) * 100 if results["total"] > 0 else 0
        
        return results