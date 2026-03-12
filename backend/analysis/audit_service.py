"""
Audit Trail Service with PDF export.

Tracks the full lifecycle of events:
  Detection → Action → Resolution → Outcome

Supports PDF generation for download and re-ingestion into the system.
"""
import io
import json
from datetime import datetime
from typing import Dict, List, Optional
from database import db
from utils.logger import SystemLogger

logger = SystemLogger(module_name="audit_trail")


def record_event_detection(zone: str, event_type: str, severity: str,
                           description: str, detection_method: str = "bayesian_network") -> int:
    """Record that an event was detected."""
    entry_id = db.store_audit_entry({
        "zone": zone,
        "event_type": event_type,
        "event_description": description,
        "event_timestamp": datetime.now().isoformat(),
        "severity": severity,
        "detection_method": detection_method,
    })
    logger.log(f"Audit: event detected — {event_type} in {zone} (id={entry_id})")
    return entry_id


def record_action(entry_id: int, action: str, operator: str = "system"):
    """Record that an action was taken on an audit entry."""
    db.update_audit_entry(entry_id, {
        "action_taken": action,
        "action_timestamp": datetime.now().isoformat(),
        "operator": operator,
    })
    logger.log(f"Audit: action recorded on entry {entry_id}")


def record_resolution(entry_id: int, description: str, outcome: str = "resolved"):
    """Record that an event was resolved."""
    db.update_audit_entry(entry_id, {
        "resolved": 1,
        "resolution_timestamp": datetime.now().isoformat(),
        "resolution_description": description,
        "outcome": outcome,
    })
    logger.log(f"Audit: resolution recorded on entry {entry_id}")


def add_note(entry_id: int, note: str):
    """Add a note to an existing audit entry."""
    # Get current notes and append
    entries = db.query_rows("audit_trail", where="id = ?", params=(entry_id,), limit=1)
    if not entries:
        return
    existing = entries[0].get("notes", "") or ""
    new_notes = f"{existing}\n[{datetime.now().isoformat()}] {note}" if existing else f"[{datetime.now().isoformat()}] {note}"
    db.update_audit_entry(entry_id, {"notes": new_notes})


def get_trail(zone: Optional[str] = None, limit: int = 100) -> List[Dict]:
    """Get audit trail entries."""
    return db.get_audit_trail(zone=zone, limit=limit)


def get_trail_range(zone: str, start: str, end: str) -> List[Dict]:
    """Get audit trail for a date range."""
    return db.get_audit_trail_range(zone, start, end)


def generate_pdf(entries: List[Dict], zone: Optional[str] = None) -> bytes:
    """
    Generate a PDF report of audit trail entries.
    Uses a simple text-based PDF generation (no external dependencies beyond stdlib).
    """
    # Build PDF manually — minimal valid PDF
    pdf = _SimplePDF()
    pdf.add_page()

    # Title
    title = f"AUDIT TRAIL REPORT — {zone or 'ALL ZONES'}"
    pdf.set_font_size(18)
    pdf.add_text(title, bold=True)
    pdf.add_text(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    pdf.add_text(f"Total Entries: {len(entries)}")
    pdf.add_text("")
    pdf.add_line()

    for i, entry in enumerate(entries):
        pdf.set_font_size(12)
        pdf.add_text(f"Entry #{i + 1}", bold=True)
        pdf.set_font_size(10)
        pdf.add_text(f"  Zone: {entry.get('zone', 'N/A')}")
        pdf.add_text(f"  Event: {entry.get('event_type', 'N/A')} — {entry.get('event_description', '')}")
        pdf.add_text(f"  Severity: {entry.get('severity', 'N/A')}")
        pdf.add_text(f"  Detected: {entry.get('event_timestamp', 'N/A')} via {entry.get('detection_method', 'N/A')}")

        if entry.get("action_taken"):
            pdf.add_text(f"  Action: {entry['action_taken']} at {entry.get('action_timestamp', '')}")
            pdf.add_text(f"  Operator: {entry.get('operator', 'system')}")

        if entry.get("resolved"):
            pdf.add_text(f"  Resolved: {entry.get('resolution_timestamp', '')} — {entry.get('resolution_description', '')}")
            pdf.add_text(f"  Outcome: {entry.get('outcome', 'N/A')}")
        else:
            pdf.add_text("  Status: UNRESOLVED")

        if entry.get("notes"):
            pdf.add_text(f"  Notes: {entry['notes'][:200]}")

        pdf.add_text("")
        pdf.add_line()

        # Check if we need a new page
        if pdf.current_y > 700:
            pdf.add_page()

    return pdf.build()


def ingest_audit_pdf_data(data: Dict) -> int:
    """
    Re-ingest audit data back into the system.
    This allows learning from past resolved events.
    For example: traffic was caused by flood → resolved by building a flyover
    → system learns that flyover prevents flood-related traffic.
    """
    entry_id = db.store_audit_entry({
        "zone": data["zone"],
        "event_type": data["event_type"],
        "event_description": data.get("event_description", ""),
        "event_timestamp": data["event_timestamp"],
        "severity": data.get("severity", "medium"),
        "detection_method": data.get("detection_method", "historical_import"),
        "action_taken": data.get("action_taken"),
        "action_timestamp": data.get("action_timestamp"),
        "resolved": 1 if data.get("resolved") else 0,
        "resolution_timestamp": data.get("resolution_timestamp"),
        "resolution_description": data.get("resolution_description"),
        "outcome": data.get("outcome"),
        "operator": data.get("operator", "historical_import"),
        "notes": data.get("notes"),
    })
    logger.log(f"Audit: ingested historical entry for {data['zone']} — {data['event_type']} (id={entry_id})")
    return entry_id


# ── Minimal PDF Generator ────────────────────────────────────────────────────

class _SimplePDF:
    """Generates a basic valid PDF without external libraries."""

    def __init__(self):
        self.pages = []
        self.current_page_content = []
        self.current_y = 50
        self.font_size = 10
        self.page_count = 0

    def add_page(self):
        if self.current_page_content:
            self.pages.append(self.current_page_content)
        self.current_page_content = []
        self.current_y = 50
        self.page_count += 1

    def set_font_size(self, size: int):
        self.font_size = size

    def add_text(self, text: str, bold: bool = False):
        # Escape PDF special characters
        safe_text = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        if bold:
            cmd = f"BT /F2 {self.font_size} Tf 50 {800 - self.current_y} Td ({safe_text}) Tj ET"
        else:
            cmd = f"BT /F1 {self.font_size} Tf 50 {800 - self.current_y} Td ({safe_text}) Tj ET"
        self.current_page_content.append(cmd)
        self.current_y += self.font_size + 4

    def add_line(self):
        cmd = f"50 {800 - self.current_y} m 550 {800 - self.current_y} l S"
        self.current_page_content.append(cmd)
        self.current_y += 8

    def build(self) -> bytes:
        if self.current_page_content:
            self.pages.append(self.current_page_content)

        objects = []
        # Obj 1: Catalog
        objects.append(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n")
        # Obj 2: Pages
        page_refs = " ".join(f"{i + 4} 0 R" for i in range(len(self.pages)))
        objects.append(f"2 0 obj\n<< /Type /Pages /Kids [{page_refs}] /Count {len(self.pages)} >>\nendobj\n".encode())
        # Obj 3: Font
        objects.append(b"3 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n")

        # Font bold
        font_bold_obj = len(objects) + 1
        objects.append(f"{font_bold_obj} 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>\nendobj\n".encode())

        next_obj = font_bold_obj + 1

        page_obj_ids = []
        content_obj_ids = []

        for page_content in self.pages:
            content_str = "\n".join(page_content)
            content_bytes = content_str.encode("latin-1", errors="replace")

            content_id = next_obj
            page_id = next_obj + 1
            next_obj += 2

            objects.append(f"{content_id} 0 obj\n<< /Length {len(content_bytes)} >>\nstream\n".encode() + content_bytes + b"\nendstream\nendobj\n")
            objects.append(f"{page_id} 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 842] /Contents {content_id} 0 R /Resources << /Font << /F1 3 0 R /F2 {font_bold_obj} 0 R >> >> >>\nendobj\n".encode())
            page_obj_ids.append(page_id)
            content_obj_ids.append(content_id)

        # Fix Pages kids to point to page objects
        page_refs = " ".join(f"{pid} 0 R" for pid in page_obj_ids)
        objects[1] = f"2 0 obj\n<< /Type /Pages /Kids [{page_refs}] /Count {len(self.pages)} >>\nendobj\n".encode()

        # Build PDF
        buf = io.BytesIO()
        buf.write(b"%PDF-1.4\n")
        offsets = []
        for obj in objects:
            offsets.append(buf.tell())
            buf.write(obj)

        xref_start = buf.tell()
        buf.write(b"xref\n")
        buf.write(f"0 {len(objects) + 1}\n".encode())
        buf.write(b"0000000000 65535 f \n")
        for off in offsets:
            buf.write(f"{off:010d} 00000 n \n".encode())

        buf.write(b"trailer\n")
        buf.write(f"<< /Size {len(objects) + 1} /Root 1 0 R >>\n".encode())
        buf.write(b"startxref\n")
        buf.write(f"{xref_start}\n".encode())
        buf.write(b"%%EOF\n")

        return buf.getvalue()
