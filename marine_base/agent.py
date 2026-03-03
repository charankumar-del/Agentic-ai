import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

# Loads the API key from your .env file for security
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

def generate_compliance_report(detection_table, risk_level):
    if not detection_table:
        return "No anomalies detected. Subsea infrastructure is assessed as SAFE."

    # Initialize the LLM with low temperature for engineering accuracy
    llm = ChatOpenAI(model="gpt-4o", temperature=0.1)
    
    # Formats YOLO data: [[class, conf], ...] -> "Class (Confidence %)"
    detections_str = ", ".join([f"{item[0]} ({item[1]*100:.1f}%)" for item in detection_table])
    
    prompt_template = """
    You are a Lead Subsea Integrity Consultant. 
    The offshore ROV edge-node just detected the following pipeline anomalies: 
    {detections}
    
    The initial vision model assigned a risk level of: {risk}
    
    Using standard DNV-OS-F101 and API pipeline regulations, generate a professional, 
    board-ready repair recommendation.
    
    Format the report with:
    1. Executive Severity Assessment
    2. Legally Mandated Next Steps (cite DNV/API standards)
    3. Estimated Intervention Urgency
    """
    
    prompt = PromptTemplate(
        input_variables=["detections", "risk"],
        template=prompt_template
    )
    
    chain = prompt | llm
    
    try:
        response = chain.invoke({"detections": detections_str, "risk": risk_level})
        return response.content
    except Exception as e:
        return f"Agent Error: {str(e)}"
