"""
LangForge — Servidor de generación de código Java
Ejecutar: python server.py
Puerto:   8000
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import json
import time

# ── Intentar importar ollama; si no está disponible, usar modo mock ──
try:
    from ollama import chat as ollama_chat
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

app = FastAPI(title="LangForge Code Generator")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

MODEL = "qwen2.5-coder:7b"

# ══════════════════════════════════════════════════════
#  SCHEMAS
# ══════════════════════════════════════════════════════

class ExprBinariaRequest(BaseModel):
    descripcion: str
    tipo_inputs: list[str] = ["num", "num"]
    tipo_output: str = "num"

class ExprUnariaRequest(BaseModel):
    descripcion: str
    tipo_input: str = "num"
    tipo_output: str = "num"

class InstruccionRequest(BaseModel):
    descripcion: str
    estructura: list[str] = []
    tipos: list[str] = []


# ══════════════════════════════════════════════════════
#  SYSTEM PROMPTS
# ══════════════════════════════════════════════════════

SYSTEM_BINARIA = """
You are an expert at converting user requests into Java execute() method bodies for BINARY operators.

Your task is to generate a JSON object with a single key "java_code" containing only the body of an execute() method that returns an Integer.

Strict rules:
1. To get the value of the left (first) expression:  left_value  (already an Integer)
2. To get the value of the right (second) expression: right_value (already an Integer)
3. User references are 1-indexed: "first expression" -> left_value, "second expression" -> right_value
4. The code must ONLY contain the body of: public Integer execute() { ... }
5. Do NOT include markdown, explanations, or the method signature.
6. Return ONLY valid JSON: {"java_code": "..."}
7. In case the return is a bool YOU MUST CONVERT IT TO AN INT. You can use the following:
    return result ? 1 : 0;
    where result is the computed expression

Examples:
User: Return the maximum between the first expression and the second
Output: {"java_code": "return Math.max(left_value, right_value);"}

User: Return the first expression to the power of the second
Output: {"java_code": "return (int) Math.pow((double) left_value, (double) right_value);"}

User: Return the sum of both expressions
Output: {"java_code": "return left_value + right_value;"}
"""

SYSTEM_UNARIA = """
You are an expert at converting user requests into Java execute() method bodies for UNARY operators.

Your task is to generate a JSON object with a single key "java_code" containing only the body of an execute() method that returns an Integer.

Strict rules:
1. To get the value of the (only) expression: value  (already an Integer)
2. The code must ONLY contain the body of: public Integer execute() { ... }
3. Do NOT include markdown, explanations, or the method signature.
4. Return ONLY valid JSON: {"java_code": "..."}
5. In case the return is a bool YOU MUST CONVERT IT TO AN INT. You can use the following:
    return result ? 1 : 0;
    where result is the computed expression

Examples:
User: Return the absolute value of the expression
Output: {"java_code": "return Math.abs(value);"}

User: Return the expression multiplied by itself
Output: {"java_code": "return value * value;"}
"""

SYSTEM_INSTRUCCION = """
You are an expert at converting user requests into Java execute() method bodies for language INSTRUCTIONS (statements).

Your task is to generate a JSON object with a single key "java_code" containing only the body of a void execute() method.

Strict rules:
1. To execute the i-th expression:             expr_i.execute()
2. To get the value of the i-th identifier:    iden_i.getValue()
3. To set the value of the i-th identifier:    iden_i.setValue(v)
4. To execute one instruction:                 instr_i.execute()
5. To execute the i-th list of instructions:
   for (Instruction instr : list_instr_i) { instr.execute(); }
6. User references are 1-indexed; generated variables MUST be 0-indexed.
   "first expression" -> expr_0,  "second expression" -> expr_1, etc.
7. The code must ONLY contain the body of: public void execute() { ... }
8. Do NOT include markdown, explanations, or the method signature.
9. Return ONLY valid JSON: {"java_code": "..."}

Examples:
User: Print the value of the first expression
Output: {"java_code": "System.out.println(expr_0.execute());"}

User: Assign to the first identifier the value of the first expression. Then repeat while the identifier is less than the second expression: execute the first list of instructions and increment the identifier by 1.
Output: {"java_code": "iden_0.setValue(expr_0.execute());\\nwhile (iden_0.getValue() < expr_1.execute()) {\\n    for (Instruction instr : list_instr_0) { instr.execute(); }\\n    iden_0.setValue(iden_0.getValue() + 1);\\n}"}
"""


# ══════════════════════════════════════════════════════
#  STREAMING GENERATOR
# ══════════════════════════════════════════════════════

def stream_generation(system_prompt: str, user_message: str):
    """
    Yields SSE events:
      data: {"type": "progress", "text": "..."}   — intermediate tokens
      data: {"type": "result",   "code": "..."}   — final extracted Java code
      data: {"type": "error",    "message": "..."}
    """

    if not OLLAMA_AVAILABLE:
        # ── Mock mode when ollama is not installed ──
        yield f"data: {json.dumps({'type': 'progress', 'text': 'Ollama no disponible — modo demo'})}\n\n"
        time.sleep(0.4)
        mock_code = "return left_value + right_value; // demo mode"
        for ch in mock_code:
            yield f"data: {json.dumps({'type': 'progress', 'text': ch})}\n\n"
            time.sleep(0.02)
        yield f"data: {json.dumps({'type': 'result', 'code': mock_code})}\n\n"
        return

    try:
        yield f"data: {json.dumps({'type': 'progress', 'text': '⟳ Conectando con el modelo...'})}\n\n"

        # Non-streaming call (ollama structured output doesn't stream well)
        # We simulate progress with a spinner while waiting
        import threading

        result_holder = {}
        error_holder = {}

        def call_ollama():
            try:
                response = ollama_chat(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": user_message}
                    ],
                    format={
                        "type": "object",
                        "properties": {
                            "java_code": {"type": "string"}
                        },
                        "required": ["java_code"]
                    },
                )
                result_holder['content'] = response.message.content
            except Exception as e:
                error_holder['msg'] = str(e)

        thread = threading.Thread(target=call_ollama)
        thread.start()

        # Send progress ticks while waiting
        dots = ["⠋","⠙","⠹","⠸","⠼","⠴","⠦","⠧","⠇","⠏"]
        i = 0
        while thread.is_alive():
            yield f"data: {json.dumps({'type': 'progress', 'text': f'{dots[i % len(dots)]} Generando código Java...'})}\n\n"
            i += 1
            time.sleep(0.2)
            thread.join(timeout=0.2)

        if 'msg' in error_holder:
            yield f"data: {json.dumps({'type': 'error', 'message': error_holder['msg']})}\n\n"
            return

        raw = result_holder.get('content', '{}')

        # Parse the JSON result
        try:
            parsed = json.loads(raw)
            code = parsed.get('java_code', raw)
        except json.JSONDecodeError:
            # Fallback: try to extract java_code manually
            code = raw.strip()

        # Stream the final code character by character for visual effect
        yield f"data: {json.dumps({'type': 'progress', 'text': '✓ Código generado — procesando...'})}\n\n"
        time.sleep(0.1)

        # Stream the code for display
        for ch in code:
            yield f"data: {json.dumps({'type': 'stream', 'char': ch})}\n\n"
        
        yield f"data: {json.dumps({'type': 'result', 'code': code})}\n\n"

    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"


# ══════════════════════════════════════════════════════
#  ENDPOINTS
# ══════════════════════════════════════════════════════

@app.get("/health")
def health():
    return {"status": "ok", "ollama": OLLAMA_AVAILABLE, "model": MODEL}


@app.post("/generate/expr-binaria")
def generate_expr_binaria(req: ExprBinariaRequest):
    user_msg = (
        f"{req.descripcion}\n\n"
        f"Input types: {', '.join(req.tipo_inputs)}\n"
        f"Output type: {req.tipo_output}"
    )
    return StreamingResponse(
        stream_generation(SYSTEM_BINARIA, user_msg),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )


@app.post("/generate/expr-unaria")
def generate_expr_unaria(req: ExprUnariaRequest):
    user_msg = (
        f"{req.descripcion}\n\n"
        f"Input type: {req.tipo_input}\n"
        f"Output type: {req.tipo_output}"
    )
    return StreamingResponse(
        stream_generation(SYSTEM_UNARIA, user_msg),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )


@app.post("/generate/instruccion")
def generate_instruccion(req: InstruccionRequest):
    estructura_str = " ".join(req.estructura) if req.estructura else "(not specified)"
    tipos_str = ", ".join(req.tipos) if req.tipos else "not specified"
    user_msg = (
        f"{req.descripcion}\n\n"
        f"Syntax structure tokens: {estructura_str}\n"
        f"Expression types (in order): {tipos_str}"
    )
    return StreamingResponse(
        stream_generation(SYSTEM_INSTRUCCION, user_msg),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )


# ══════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    print("╔══════════════════════════════════════════╗")
    print("║   LangForge Code Generator — servidor    ║")
    print(f"║   Ollama disponible: {str(OLLAMA_AVAILABLE):<19} ║")
    print(f"║   Modelo: {MODEL:<31} ║")
    print("║   Escuchando en http://localhost:8000    ║")
    print("╚══════════════════════════════════════════╝")
    uvicorn.run(app, host="0.0.0.0", port=8000)
