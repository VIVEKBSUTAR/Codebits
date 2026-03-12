"""
LLM-based recommendation engine.
Provides natural-language operational recommendations based on event context.
Supports OpenAI GPT and Google Gemini via API keys.
Falls back to a rule-based engine when no API key is available.
"""
import os
import json
import httpx
from datetime import datetime
from typing import Dict, Optional, List
from database import db
from utils.logger import SystemLogger

logger = SystemLogger(module_name="llm_engine")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")


def _build_prompt(zone: str, event_context: Dict) -> str:
    """Build a structured prompt for the LLM."""
    events = event_context.get("events", [])
    risks = event_context.get("risks", {})
    weather = event_context.get("weather", {})

    prompt = f"""You are an AI city operations advisor for Pune, India.

Zone: {zone}

Current Situation:
- Active events: {json.dumps(events, default=str)}
- Risk levels: {json.dumps(risks, default=str)}
- Weather: {json.dumps(weather, default=str)}

Based on this situation, provide:
1. **Immediate Actions** — what should be done right now (2-3 bullet points)
2. **Root Cause Assessment** — what is likely causing this situation
3. **Resource Deployment** — which resources to deploy and where
4. **Escalation Risk** — what could happen if no action is taken
5. **Recommended Priority** — HIGH / MEDIUM / LOW

Keep the response concise and actionable. Focus on practical city operations."""
    return prompt


async def _call_openai(prompt: str) -> Optional[Dict]:
    """Call OpenAI GPT API."""
    if not OPENAI_API_KEY:
        return None
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
                json={
                    "model": "gpt-4o-mini",
                    "messages": [
                        {"role": "system", "content": "You are an expert city operations AI advisor."},
                        {"role": "user", "content": prompt},
                    ],
                    "max_tokens": 800,
                    "temperature": 0.3,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            text = data["choices"][0]["message"]["content"]
            tokens = data.get("usage", {}).get("total_tokens", 0)
            return {"response": text, "model": "gpt-4o-mini", "tokens_used": tokens}
    except Exception as e:
        logger.log(f"OpenAI call failed: {e}")
        return None


async def _call_gemini(prompt: str) -> Optional[Dict]:
    """Call Google Gemini API."""
    if not GEMINI_API_KEY:
        return None
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                url,
                json={
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {"maxOutputTokens": 800, "temperature": 0.3},
                },
            )
            resp.raise_for_status()
            data = resp.json()
            text = data["candidates"][0]["content"]["parts"][0]["text"]
            tokens = data.get("usageMetadata", {}).get("totalTokenCount", 0)
            return {"response": text, "model": "gemini-2.0-flash", "tokens_used": tokens}
    except Exception as e:
        logger.log(f"Gemini call failed: {e}")
        return None


def _rule_based_recommendation(zone: str, event_context: Dict) -> Dict:
    """Fallback rule-based recommendation when no LLM API key is configured."""
    events = event_context.get("events", [])
    risks = event_context.get("risks", {})

    actions = []
    root_cause = "Unknown — configure LLM API keys for detailed analysis"
    resources = []
    priority = "MEDIUM"

    event_types = [e.get("event_type", "") for e in events]
    risk_values = risks

    # Rule-based logic
    if "rainfall" in event_types or risks.get("flooding", 0) > 0.5:
        actions.append("Activate emergency pumping stations in low-lying areas")
        actions.append("Issue public advisory for waterlogging-prone roads")
        resources.append("Deploy 2 pump units to drainage hotspots")
        root_cause = "Heavy rainfall exceeding drainage capacity"
        priority = "HIGH"

    if "accident" in event_types or risks.get("traffic_congestion", 0) > 0.6:
        actions.append("Deploy traffic police to affected intersections")
        actions.append("Activate alternate route signage and navigation alerts")
        resources.append("Dispatch 3 traffic officers")
        root_cause = root_cause if "rainfall" in event_types else "Road incident causing congestion cascade"
        priority = "HIGH"

    if "construction" in event_types:
        actions.append("Verify construction permits and safety compliance")
        actions.append("Implement temporary traffic diversions")
        resources.append("Assign 1 traffic management unit")

    if risks.get("emergency_delay", 0) > 0.4:
        actions.append("Pre-position ambulance at zone boundary")
        actions.append("Clear emergency corridor on main arterial road")
        resources.append("Alert nearest hospital for standby")
        priority = "HIGH"

    if not actions:
        actions = ["Continue monitoring — no immediate action required"]
        resources = ["No additional resources needed"]
        priority = "LOW"

    escalation = "Low risk of further cascading" if priority == "LOW" else \
                 "Moderate risk: situation may worsen without intervention within 30 minutes" if priority == "MEDIUM" else \
                 "High risk: cascading failures likely within 15 minutes without intervention"

    response_text = f"""## Immediate Actions
{chr(10).join('- ' + a for a in actions)}

## Root Cause Assessment
{root_cause}

## Resource Deployment
{chr(10).join('- ' + r for r in resources)}

## Escalation Risk
{escalation}

## Recommended Priority
**{priority}**"""

    return {
        "response": response_text,
        "model": "rule-based-fallback",
        "tokens_used": 0,
    }


async def get_recommendation(zone: str, event_context: Dict) -> Dict:
    """
    Get LLM recommendation for an event situation.
    Tries OpenAI → Gemini → Rule-based fallback.
    Stores result in database.
    """
    prompt = _build_prompt(zone, event_context)

    # Try LLM providers in order
    result = await _call_openai(prompt)
    if not result:
        result = await _call_gemini(prompt)
    if not result:
        result = _rule_based_recommendation(zone, event_context)

    # Store to database
    try:
        db.store_llm_log({
            "zone": zone,
            "event_context": event_context,
            "prompt": prompt,
            "response": result["response"],
            "model": result["model"],
            "tokens_used": result.get("tokens_used", 0),
        })
    except Exception as e:
        logger.log(f"Failed to store LLM log: {e}")

    return {
        "zone": zone,
        "recommendation": result["response"],
        "model_used": result["model"],
        "tokens_used": result.get("tokens_used", 0),
        "timestamp": datetime.now().isoformat(),
    }
