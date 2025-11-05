# LangChain 1.0+ uses LCEL (LangChain Expression Language)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.documents import Document
from operator import itemgetter


def create_qa_chain(llm, vectorstore):
    """Create a RAG chain using LCEL (LangChain Expression Language) for LangChain 1.0+"""
    
    # Create retriever - optimized for SPEED
    retriever = vectorstore.as_retriever(
        search_type="similarity",  # Faster than MMR
        search_kwargs={
            "k": 8,  # Fewer chunks = faster (was 20)
        }
    )
    
    # AI-enhanced system prompt - Natural but accurate
    system_prompt = """You are an experienced academic advisor at SRMIST, helping students understand the Academic Regulations 2021. Your knowledge comes from data-1.pdf.

YOUR ROLE:
- Be helpful, friendly, and conversational (like a real advisor)
- Explain regulations in clear, student-friendly terms
- Provide comprehensive answers with all relevant details
- Use the EXACT information from the context below - no invention
- Add helpful context and explanations to make rules understandable

HOW TO ANSWER:
1. Start with a clear, direct answer - address their specific question first
2. Explain thoroughly - break down complex rules into simple terms
3. Provide context - help them understand WHY the rule exists
4. Include ALL relevant details - numbers, dates, conditions, exceptions
5. Use clean, simple formatting:
   - Numbered lists for step-by-step processes
   - Bullet points (•) for lists of items
   - Keep it natural and easy to read
   - NO bold text unless absolutely critical

6. Cite sources clearly: 
   - After each factual point: [Page X]
   - At the end: Sources: Pages X, Y, Z

STYLE:
- Conversational but professional
- Helpful and encouraging
- Define acronyms naturally (e.g., "You'll need a minimum CGPA - that's your Cumulative Grade Point Average...")
- Use "you" and "your" to make it personal
- Anticipate follow-up questions and address them
- Keep formatting minimal and clean

EXAMPLE GOOD ANSWER:
"Great question! The minimum CGPA required for graduation is 5.0 out of 10 [Page 12].

Here's what you need to know:

During Your Program:
• Maintain at least 5.0 CGPA across all semesters [Page 12]
• Clear all courses with passing grades [Page 15]
• Complete the required 160 credits [Page 8]

Important Points:
• If your CGPA drops below 5.0, you'll be placed on academic probation [Page 24]
• You can improve through supplementary exams [Page 45]
• Final degree classification depends on your overall CGPA [Page 50]

Sources: Pages 8, 12, 15, 24, 45, 50

Need more details about any of these requirements? Feel free to ask!"

CRITICAL RULES:
- Search the ENTIRE context thoroughly before saying information is unavailable
- Look for synonyms and related terms (e.g., "moderation" might be "review committee")
- Extract ALL relevant details from the context
- If truly NOT in context: "I couldn't find this information in the regulations. Please contact the academic office."
- NEVER make up information
- If you find partial information, provide it and note what's missing
- Keep formatting clean and minimal - NO excessive bold text

Context from SRMIST Academic Regulations 2021:
{context}

Current Date: {current_date}

Now help this student with their question:"""

    
    # Create prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{input}")
    ])
    
    # Helper function to format documents
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    # Build the LCEL chain
    # Step 1: Retrieve documents
    # Step 2: Format context and pass through
    # Step 3: Generate answer
    chain = (
        RunnableParallel({
            "context": itemgetter("input") | retriever | format_docs,
            "input": itemgetter("input"),
            "chat_history": itemgetter("chat_history"),
            "current_date": itemgetter("current_date"),
        })
        | prompt
        | llm
        | StrOutputParser()
    )
    
    # Wrap to match expected output format
    def chain_with_context(inputs):
        result = chain.invoke(inputs)
        # Retrieve docs for metadata
        docs = retriever.invoke(inputs["input"])
        return {
            "answer": result,
            "context": docs,
            "input": inputs["input"]
        }
    
    return chain_with_context


def check_answer_type_chain(llm):
    """Check if the answer is related to academic topics (for SRMIST regulations)"""
    check_answer_prompt = """You are an automated relevance classifier. Decide whether the following pair (User Query, RAG Answer) concerns academic regulations contained in data-1.pdf (SRMIST Academic Regulations 2021).

Academic topics include: grading, CIE, SEE, CGPA, credits, course registration, vertical progression, internships, degree requirements, examination rules, moderation committee, re-evaluation, supplementary exams, attendance, academic calendar and similar academic-regulation topics.

User Query: {user_query}

RAG Answer: {rag_answer}

Respond with ONLY YES if the query+answer are related to these academic regulation topics, or ONLY NO if they are not. Nothing else.
"""
    
    prompt = ChatPromptTemplate.from_template(check_answer_prompt)
    checker_chain = prompt | llm | StrOutputParser()
    return checker_chain


def create_checker_chain(llm):
    """Chain to verify and correct RAG answers for academic queries with citations"""
    checker_prompt = """You are a verifier that must check and, if necessary, correct a RAG-produced answer so it strictly aligns with the content of data-1.pdf (SRMIST Academic Regulations 2021).

Tasks:
1. Verify the RAG Answer only using the provided RAG context (do not use outside knowledge).

2. If the RAG Answer is fully supported by the document, return it unchanged but append the file citation token where factual claims are made: :contentReference[oaicite:3]{{index=3}}. If a specific section or page was provided in context metadata, include it after the citation (e.g., (sec. 4.2) or (p. 35)).

3. If the RAG Answer contains incorrect, extraneous, or unsupported claims, produce a corrected, concise answer strictly supported by the document and append the file citation token: :contentReference[oaicite:4]{{index=4}}.

4. If the question cannot be answered from the document, return exactly: "The information is not available in the provided document."

5. Maintain clear, neutral language and preserve any numeric/date format rules (numbers in numeric format, dates as DD-MM-YYYY).


User Query: {user_query}

RAG Answer: {rag_answer}
"""
    
    prompt = ChatPromptTemplate.from_template(checker_prompt)
    checker_chain = prompt | llm | StrOutputParser()
    return checker_chain


# Note: The following probationary checker functions are from the original HR system
# They are kept for backwards compatibility but are not used in the academic RAG system

def probationary_checker(llm):
    """Legacy function - not used in academic RAG"""
    pass

def probationary_data_modifier(llm):
    """Legacy function - not used in academic RAG"""
    pass
