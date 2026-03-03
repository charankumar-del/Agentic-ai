import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic

# Newer versions of LangChain expose PromptTemplate from langchain_core.prompts,
# not langchain.prompts. This import keeps the code compatible.
from langchain_core.prompts import PromptTemplate

# Loads environment variables from your .env file so ANTHROPIC_API_KEY is available
load_dotenv()

def generate_compliance_report(detection_table, risk_level):
    if not detection_table:
        return "No anomalies detected. Subsea infrastructure is assessed as SAFE."

    # Initialize Anthropic Claude via LangChain with low temperature for engineering accuracy
    llm = ChatAnthropic(model="claude-3-5-sonnet-latest", temperature=0.1)
    
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
        # If the OpenAI key is out of quota or the API is unavailable,
        # fall back to a rule-based report so the app still works.
        msg = str(e)
        quota_issue = (
            "insufficient_quota" in msg
            or "You exceeded your current quota" in msg
            or "429" in msg
        )

        severity_text = {
            "HIGH": "High-risk structural anomaly profile detected. Immediate intervention is recommended.",
            "MEDIUM": "Moderate anomaly profile detected. Near-term remediation planning is recommended.",
            "LOW": "Low-level anomalies detected. Monitor during the next scheduled inspection window.",
            "SAFE": "No actionable anomalies detected. System is within acceptable integrity thresholds.",
        }.get(str(risk_level).upper(), "Risk level could not be determined from the input.")

        detections_readable = detections_str if detections_str else "No model detections available."

        fallback_header = "EXECUTIVE SEVERITY ASSESSMENT\n"
        fallback_body = (
            f"- Model-assigned visual risk level: {risk_level}\n"
            f"- Interpreted severity summary: {severity_text}\n\n"
            "LEGALLY MANDATED NEXT STEPS (REFERENCE ONLY)\n"
            "- Ensure inspection and maintenance planning aligns with the intent of DNV-OS-F101 and API pipeline integrity guidelines.\n"
            "- Log this inspection, associated media, and derived risk classification into your internal integrity management system.\n"
            "- If corrosion, coating damage, or debris accumulation is present, plan follow-up NDT/inspection as per your asset-specific strategy.\n\n"
            "ESTIMATED INTERVENTION URGENCY\n"
            "- HIGH: Plan corrective action in the nearest feasible maintenance window; consider temporary derating if applicable.\n"
            "- MEDIUM: Address in the next scheduled subsea campaign; continue monitoring for trend escalation.\n"
            "- LOW/SAFE: No immediate intervention required; retain for historical traceability.\n\n"
            "DETECTION SNAPSHOT\n"
            f"- Raw detections: {detections_readable}\n"
        )

        if quota_issue:
            notice = (
                "NOTE: LLM-based compliance reasoning was skipped because the configured "
                "OpenAI API key has insufficient quota. The above summary is generated "
                "locally using deterministic rules.\n"
            )
        else:
            notice = (
                "NOTE: LLM-based compliance reasoning was unavailable due to an error. "
                "The above summary is generated locally using deterministic rules.\n"
                f"(Underlying error: {msg})\n"
            )

        return fallback_header + "\n" + fallback_body + "\n" + notice
