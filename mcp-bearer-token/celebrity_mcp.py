# Celebrity Chat MCP Server (WhatsApp persona chat, OpenAI Responses API)
# Flow:
# 1) User provides celebrity name via set_celebrity_name
# 2) Server builds a persona system prompt
# 3) send_message uses that prompt + short user history for all future turns
#
# Env:
#   AUTH_TOKEN=...          (same as starter)
#   MY_NUMBER=919876543210  (same as starter)
#   OPENAI_API_KEY=sk-...   (OpenAI)
#   OPENAI_MODEL=gpt-5-mini | gpt-5-nano | o4-mini  (optional; defaults in that order)

import asyncio, os, json
from typing import Annotated, Optional
from datetime import datetime
from dotenv import load_dotenv

from fastmcp import FastMCP
from fastmcp.server.auth.providers.bearer import BearerAuthProvider, RSAKeyPair
from mcp.server.auth.provider import AccessToken
from mcp import ErrorData, McpError
from mcp.types import TextContent, INVALID_PARAMS, INTERNAL_ERROR
from pydantic import Field, BaseModel

# --- OpenAI SDK (Responses API) ---
try:
    from openai import OpenAI
except Exception as e:
    raise RuntimeError("Please install openai>=1.0:  pip install openai") from e

# --- Env ---
load_dotenv()
TOKEN = os.environ.get("AUTH_TOKEN")
MY_NUMBER = os.environ.get("MY_NUMBER")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL")  # optional override

assert TOKEN, "Please set AUTH_TOKEN in your .env file"
assert MY_NUMBER is not None, "Please set MY_NUMBER in your .env file"
assert OPENAI_API_KEY, "Please set OPENAI_API_KEY in your .env file"

# Preferred models: gpt-5-mini → gpt-5-nano → o4-mini
_PREFERRED_MODELS = [m for m in [OPENAI_MODEL, "gpt-5-mini", "gpt-5-nano", "o4-mini"] if m]

# --- Auth (same pattern as starter) ---
class SimpleBearerAuthProvider(BearerAuthProvider):
    def __init__(self, token: str):
        k = RSAKeyPair.generate()
        super().__init__(public_key=k.public_key, jwks_uri=None, issuer=None, audience=None)
        self.token = token

    async def load_access_token(self, token: str) -> AccessToken | None:
        if token == self.token:
            return AccessToken(token=token, client_id="celebrity-client", scopes=["*"], expires_at=None)
        return None

mcp = FastMCP("Celebrity Chat MCP Server", auth=SimpleBearerAuthProvider(TOKEN))

# --- In-memory per-user state ---
# { puch_user_id: { "celebrity": "Shah Rukh Khan", "system_prompt": "...", "history": [ {role, content}, ... ] } }
STATE: dict[str, dict] = {}

def _now() -> str:
    return datetime.utcnow().isoformat()

def _user_state(puch_user_id: str) -> dict:
    if not puch_user_id:
        raise McpError(ErrorData(code=INVALID_PARAMS, message="puch_user_id is required"))
    return STATE.setdefault(puch_user_id, {"celebrity": None, "system_prompt": None, "history": []})

def _error(code, msg):
    raise McpError(ErrorData(code=code, message=msg))

# --- Rich Tool Description model (same helper as starter) ---
class RichToolDescription(BaseModel):
    description: str
    use_when: str
    side_effects: str | None = None

# --- Tool: validate (required by Puch) ---
@mcp.tool
async def validate() -> str:
    return MY_NUMBER

# --- OpenAI client ---
_oai = OpenAI(api_key=OPENAI_API_KEY)

def _pick_model() -> str:
    # You can add a health check here if you want to probe availability.
    return _PREFERRED_MODELS[0]

def _build_system_prompt(celebrity_name: str) -> str:
    # Keep it safe, WhatsApp-friendly, and clearly a persona (not the real person).
    return (
        f"You are an AI chat persona inspired by {celebrity_name}. "
        f"You are NOT the real {celebrity_name}. "
        "Default tone: friendly Hinglish (mix Hindi+English) unless the user sticks to only Hindi or only English. "
        "Avoid making factual claims about private life; avoid defamatory or unverifiable claims. "
        "No promises of personal contact or meetings. "
        "Keep most replies under ~800 characters unless the user asks for more. "
        "Use light emojis sparingly."
    )

async def _openai_respond(system_prompt: str, history: list[dict], user_text: str) -> str:
    """
    Calls OpenAI Responses API with a system prompt + short history + new message.
    History format: [{"role":"user"/"assistant","content":"..."}]
    """
    model = _pick_model()
    # keep a small rolling context for latency/cost; trim to last 10 turns
    short_hist = history[-10:]

    # Responses API accepts a simple "input" that can be a list of role messages.
    # We pass system + short history + current user message.
    # See: Quickstart / Text generation / Migrate to Responses API.  (docs cited in chat)
    try:
        resp = _oai.responses.create(
            model=model,
            input=[
                {"role": "system", "content": system_prompt},
                *short_hist,
                {"role": "user", "content": user_text},
            ],
        )
        # Text is typically in resp.output_text for the Python SDK.
        # Fallback: stitch from content items if needed.
        text = getattr(resp, "output_text", None)
        if not text:
            # compatible path: join any textual segments in the first output
            if resp and getattr(resp, "output", None):
                chunks = []
                for item in resp.output[0].content:
                    if item.get("type") == "output_text":
                        chunks.append(item.get("text", ""))
                text = "".join(chunks) if chunks else ""
        return text or "Sorry, I couldn't generate a reply right now."
    except Exception as e:
        return f"Oops—model error: {e}. Try again?"

# --- Tool descriptions ---
SET_NAME_DESC = RichToolDescription(
    description="Set the celebrity/persona name for this user and prepare the chat prompt.",
    use_when="User tells which celebrity they want to chat with.",
    side_effects="Stores the celebrity name and persona prompt in memory for this user.",
)
SEND_MSG_DESC = RichToolDescription(
    description="Send a message to the selected celebrity persona and get a Hinglish reply.",
    use_when="User continues chatting after choosing a celebrity.",
    side_effects="Appends short chat history for better continuity.",
)
GET_STATE_DESC = RichToolDescription(
    description="Get the current celebrity and brief state for this user.",
    use_when="Debugging or confirming which persona is active.",
)
RESET_DESC = RichToolDescription(
    description="Reset the user's celebrity persona and chat history.",
    use_when="User wants to start fresh or pick a new persona cleanly.",
    side_effects="Clears memory for this user.",
)

# --- Tools ---

@mcp.tool(description=SET_NAME_DESC.model_dump_json())
async def set_celebrity_name(
    puch_user_id: Annotated[str, Field(description="Puch User Unique Identifier")],
    celebrity_name: Annotated[str, Field(description="Name typed by the user (e.g., 'Shah Rukh Khan')")],
) -> list[TextContent]:
    try:
        if not celebrity_name or not celebrity_name.strip():
            _error(INVALID_PARAMS, "celebrity_name cannot be empty")
        st = _user_state(puch_user_id)
        celeb = celebrity_name.strip()
        st["celebrity"] = celeb
        st["system_prompt"] = _build_system_prompt(celeb)
        # Optionally clear history when switching personas
        st["history"] = []
        return [TextContent(type="text", text=json.dumps({"ok": True, "celebrity": celeb}))]
    except McpError:
        raise
    except Exception as e:
        _error(INTERNAL_ERROR, str(e))

@mcp.tool(description=SEND_MSG_DESC.model_dump_json())
async def send_message(
    puch_user_id: Annotated[str, Field(description="Puch User Unique Identifier")],
    text: Annotated[str, Field(description="User message to the persona")],
) -> list[TextContent]:
    try:
        if not text or not text.strip():
            _error(INVALID_PARAMS, "text cannot be empty")
        st = _user_state(puch_user_id)
        if not st.get("celebrity") or not st.get("system_prompt"):
            _error(INVALID_PARAMS, "No celebrity set yet. Call set_celebrity_name first.")
        reply = await _openai_respond(st["system_prompt"], st.get("history", []), text.strip())
        # update memory (trim)
        hist = st.get("history", [])
        hist += [{"role": "user", "content": text.strip()},
                 {"role": "assistant", "content": reply}]
        st["history"] = hist[-12:]
        return [TextContent(type="text", text=json.dumps({"reply": reply}))]
    except McpError:
        raise
    except Exception as e:
        _error(INTERNAL_ERROR, str(e))

@mcp.tool(description=GET_STATE_DESC.model_dump_json())
async def get_state(
    puch_user_id: Annotated[str, Field(description="Puch User Unique Identifier")],
) -> list[TextContent]:
    try:
        st = _user_state(puch_user_id)
        minimal = {
            "celebrity": st.get("celebrity"),
            "history_turns": len(st.get("history", [])),
            "has_prompt": bool(st.get("system_prompt")),
        }
        return [TextContent(type="text", text=json.dumps(minimal))]
    except Exception as e:
        _error(INTERNAL_ERROR, str(e))

@mcp.tool(description=RESET_DESC.model_dump_json())
async def reset_chat(
    puch_user_id: Annotated[str, Field(description="Puch User Unique Identifier")],
    keep_celebrity: Annotated[Optional[bool], Field(description="If true, keep celebrity but clear history")] = False,
) -> list[TextContent]:
    try:
        st = _user_state(puch_user_id)
        if keep_celebrity:
            celeb = st.get("celebrity")
            prompt = st.get("system_prompt")
            STATE[puch_user_id] = {"celebrity": celeb, "system_prompt": prompt, "history": []}
        else:
            STATE[puch_user_id] = {"celebrity": None, "system_prompt": None, "history": []}
        return [TextContent(type="text", text=json.dumps({"ok": True}))]
    except Exception as e:
        _error(INTERNAL_ERROR, str(e))

# --- Run MCP Server ---
async def main():
    print("🎭 Starting Celebrity Chat MCP (OpenAI) on http://0.0.0.0:8087")
    await mcp.run_async("streamable-http", host="0.0.0.0", port=8087)

if __name__ == "__main__":
    asyncio.run(main())
