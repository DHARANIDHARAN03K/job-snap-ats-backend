from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types
import os
import io
import json 

# 1. Initialize FastAPI App
app = FastAPI()

# 1B. CORS Configuration (HARDENED FOR PRODUCTION)
allowed_origins = [
    "http://localhost:3000",
    "https://job-snap-ats-backend.vercel.app", 
    "chrome-extension://fofmaacljamclpfnchaangnindfjfbjo" 
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins, 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Configure Gemini Client
try:
    # client will automatically look for GEMINI_API_KEY environment variable
    client = genai.Client()
    GEMINI_CONFIGURED = True
except Exception as e:
    # This path is hit if GEMINI_API_KEY is not set in Vercel environment
    print(f"Gemini API key not configured. Using mock data. Error: {e}") 
    GEMINI_CONFIGURED = False
    
# 3. Define the ATS API Endpoint
@app.post("/api/ats-optimize")
async def ats_optimize_resume(
    job_description: str = Form(...),
    job_title: str = Form(...), 
    resume_file: UploadFile = File(...) 
):
    # --- STEP 3A: FILE PROCESSING & VALIDATION ---
    file_mime_type = resume_file.content_type
    
    # Check for supported file types (PDF or DOCX)
    if file_mime_type and not file_mime_type.startswith(('application/pdf', 'application/vnd.openxmlformats-officedocument.wordprocessingml')):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a PDF or DOCX.")

    # --- STEP 3B: CORE GEMINI LOGIC (Conditional on configuration) ---
    if not GEMINI_CONFIGURED:
        return {
            "success": True,
            "matchScore": 75,
            "suggestions": "⚠️ MOCK DATA: Missing 'DevOps' and 'Kubernetes'. Add these to your experience section. (API Key not configured.)"
        }
        
    uploaded_file = None
    try:
        # 1. Read file content into a buffer
        file_bytes = await resume_file.read()
        file_buffer = io.BytesIO(file_bytes)
        
        # 2. Upload the file to the Gemini service - FIX: Explicitly passing mime_type
        # This resolves the 'Unknown mime type' error encountered in the Vercel environment.
        uploaded_file = client.files.upload(
            file=file_buffer,
            mime_type=file_mime_type # <--- CORRECTED LINE
        )

        # 3. Define the System Instruction for Structured Output
        system_instruction = (
            "You are an expert Applicant Tracking System (ATS) optimization engine. "
            "Analyze the provided Resume file against the Job Description. "
            "Your output MUST be a single JSON object with the keys 'matchScore' (integer percentage) and 'suggestions' (string with detailed, actionable bullet points for improvement)."
            "Do NOT include any preamble or extra text outside the JSON object."
        )

        # 4. Define the User Prompt
        prompt = (
            f"Analyze this resume against the following job description for the role: {job_title}\n\n"
            f"**Job Description:**\n{job_description}\n\n"
            "Provide the ATS Match Score and detailed, actionable suggestions for improving the resume's alignment with the job description. "
            "Your response must be a single JSON object."
        )

        # 5. Generate Content (Multimodal request with file and text)
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[
                types.Content(role='user', parts=[
                    types.Part.from_text(prompt), 
                    types.Part.from_file(uploaded_file) 
                ])
            ],
            config=types.GenerateContentConfig(
                system_instruction=system_instruction
            )
        )

        # 6. Parse the JSON response
        try:
            json_text = response.text.strip().replace("```json", "").replace("```", "")
            ai_result = json.loads(json_text)
            
            match_score = ai_result.get("matchScore", 0)
            suggestions = ai_result.get("suggestions", "No detailed suggestions provided by AI.")
            
            if not isinstance(match_score, int) or match_score < 0 or match_score > 100:
                raise ValueError("Match score is invalid.")
                
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Error parsing Gemini response: {e}")
            match_score = 50 
            suggestions = f"Error: Could not parse structured output from AI. Raw response: {response.text[:200]}..."
            
        # 7. Return the result to the Chrome Extension
        return {
            "success": True,
            "matchScore": match_score,
            "suggestions": suggestions
        }

    except Exception as e:
        print(f"Gemini API processing error: {e}")
        # Re-raise HTTPException with a specific error detail
        raise HTTPException(
            status_code=500, 
            detail=f"AI Service Error: Failed to process file. Check API keys and Vercel logs. Error: {e}"
        )
    finally:
        # CRITICAL: Delete the uploaded file after processing
        if uploaded_file:
            client.files.delete(name=uploaded_file.name)
            print(f"Successfully deleted uploaded file: {uploaded_file.name}")