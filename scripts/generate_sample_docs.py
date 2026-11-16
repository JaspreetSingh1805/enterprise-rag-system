"""
Generate realistic sample enterprise documents across PDF, DOCX, TXT, and Markdown.
"""

from pathlib import Path

import docx

SAMPLE_DIR = Path("./sample_docs")
SAMPLE_DIR.mkdir(parents=True, exist_ok=True)


def generate_hr_policy_md():
    content = """# Company HR & Leave Policy (Global Operations)

## 1. Annual Leave Entitlement
Full-time permanent employees are entitled to 25 business days of paid annual leave per calendar year. Leave accrues on a pro-rata basis at 2.08 days per completed month of service. A maximum of 5 unused annual leave days can be carried over into the following calendar year, provided they are consumed before March 31st. Any additional untaken leave beyond 5 days is forfeited automatically without financial encashment.

## 2. Sick & Medical Leave
Employees receive 12 paid sick leave days annually, granted in full on January 1st of each year. 
- Consecutive sick leave exceeding 2 business days requires an official medical certificate signed by a licensed medical practitioner.
- In cases of prolonged hospitalization, an extended medical leave of up to 30 calendar days may be approved at full pay upon review by the HR Benefits Committee.
- Sick leave cannot be carried over to subsequent years and has zero cash surrender value.

## 3. Parental & Adoption Leave
Primary caregivers are eligible for 16 weeks of fully paid parental leave following the birth or legal adoption of a child. Secondary caregivers receive 4 weeks of fully paid parental leave. To qualify, the employee must have completed at least 6 months of continuous active employment prior to the anticipated delivery or placement date.

## 4. Work From Anywhere & Remote Guidelines
Employees in eligible roles may request up to 30 work-from-anywhere days per calendar year. Requests must be submitted via the HR Portal at least 14 days in advance and require written line manager approval. Core collaboration hours remain 10:00 AM to 4:00 PM in the employee's designated home office timezone.
"""
    with open(SAMPLE_DIR / "hr_leave_policy.md", "w", encoding="utf-8") as f:
        f.write(content.strip())
    print("Generated sample_docs/hr_leave_policy.md")


def generate_security_guidelines_txt():
    content = """GLOBAL INFORMATION SECURITY & DATA PRIVACY GUIDELINES (v4.2)

1. PASSWORD REQUIREMENTS AND AUTHENTICATION
All corporate accounts must utilize passwords with a minimum length of 14 characters, containing at least one uppercase letter, one lowercase letter, one numeric digit, and one special symbol. Passwords expire every 90 days and cannot match the previous 12 passwords used. Multi-Factor Authentication (MFA) via hardware security key or corporate authenticator app is mandatory for all internal services, VPN gateways, and cloud applications. SMS-based 2FA is strictly prohibited for privileged access.

2. DEVICE ENCRYPTION AND CLEAN DESK POLICY
All company-issued laptops must have full-disk encryption enabled (BitLocker with XTS-AES 256 or FileVault) prior to connecting to the corporate network. Removable USB flash drives and unauthorized external storage devices are blocked by endpoint management agents. Employees must lock their workstation screens whenever stepping away. Physical documents containing customer Personal Identifiable Information (PII) or confidential intellectual property must be locked inside desk drawers when unattended.

3. SECURITY INCIDENT REPORTING TIMELINE
Any suspected security incident, including lost laptops, phishing email clicks, unauthorized data access, or ransomware indicators, must be reported to the Security Operations Center (SOC) at soc@company.internal within 60 minutes of detection. High-severity incidents are escalated to the Incident Response Team with a guaranteed 15-minute containment SLA.

4. REMOTE ACCESS AND VPN POLICIES
Remote network access is permitted only via the corporate WireGuard/IPsec zero-trust network access (ZTNA) tunnel. Split-tunneling is strictly disabled. Public open Wi-Fi networks (such as cafes, airports, and hotels) must only be used while connected to the full corporate VPN.
"""
    with open(SAMPLE_DIR / "security_guidelines.txt", "w", encoding="utf-8") as f:
        f.write(content.strip())
    print("Generated sample_docs/security_guidelines.txt")


def generate_employee_handbook_docx():
    doc = docx.Document()
    doc.add_heading("Employee Operations & Code of Conduct Handbook", level=0)

    doc.add_heading("1. Business Travel and Expense Reimbursements", level=1)
    doc.add_paragraph(
        "All business travel must be pre-approved by the department director at least 10 business days before booking. "
        "Employees traveling on corporate business are entitled to a daily meal per diem of $75 in domestic cities and $110 in high-cost international metro areas. "
        "Flights under 6 hours must be booked in economy class; flights exceeding 6 hours continuous travel time are eligible for premium economy or business class with VP approval. "
        "Expense receipts must be submitted through Concur within 30 days of trip conclusion."
    )

    doc.add_heading("2. Working Hours and Flexible Schedules", level=1)
    doc.add_paragraph(
        "Standard full-time working hours are 40 hours per week, typically Monday through Friday. "
        "Core hours during which all team members must be reachable on Slack or Microsoft Teams are 10:00 AM to 3:30 PM local office time. "
        "Flexible starting hours between 7:30 AM and 10:00 AM may be arranged in agreement with your immediate team lead."
    )

    doc.add_heading("3. Performance Evaluation & Compensation Review", level=1)
    doc.add_paragraph(
        "Performance reviews are conducted semi-annually in June (Mid-Year Check-in) and December (Annual Review). "
        "Ratings range from 1 (Unsatisfactory) to 5 (Exceeds Expectations). "
        "Discretionary merit bonuses and salary adjustments are tied directly to the December annual review rating and are disbursed in the February payroll cycle."
    )

    doc.save(str(SAMPLE_DIR / "employee_handbook.docx"))
    print("Generated sample_docs/employee_handbook.docx")


def generate_product_workflow_api_pdf():
    # Use pypdf to assemble a clean, valid PDF with text streams
    # Or create raw PDF stream syntax
    pdf_content = b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>
endobj
4 0 obj
<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>
endobj
5 0 obj
<< /Length 720 >>
stream
BT
/F1 16 Tf
50 720 Td
(Enterprise Payment Gateway & API Integration Workflow) Tj
/F1 12 Tf
0 -35 Td
(1. API Authentication & Token Lifecycle) Tj
0 -20 Td
(API requests must include a Bearer JWT token in the Authorization header.) Tj
0 -15 Td
(Tokens expire after 3600 seconds (1 hour). Refresh tokens are valid for 30 days.) Tj
0 -30 Td
(2. Payment Transaction Lifecycle) Tj
0 -20 Td
(Transactions transition through: CREATED -> AUTHORIZED -> CAPTURED -> SETTLED.) Tj
0 -15 Td
(Automatic settlement occurs daily at 23:59 UTC for all captured authorizations.) Tj
0 -15 Td
(Refunds can be initiated within 180 days of the original settlement date.) Tj
0 -30 Td
(3. Webhook Notifications and Idempotency) Tj
0 -20 Td
(Webhook delivery events are signed using HMAC-SHA256 with the merchant webhook secret.) Tj
0 -15 Td
(Failed webhook deliveries are retried using exponential backoff over 72 hours.) Tj
0 -15 Td
(Merchants must respond with HTTP 200 OK within 5000 milliseconds to prevent retries.) Tj
0 -30 Td
(4. Rate Limiting and Error Status Codes) Tj
0 -20 Td
(Standard API rate limit is 1200 requests per minute per API key.) Tj
0 -15 Td
(HTTP 429 Too Many Requests includes a Retry-After header with wait seconds.) Tj
ET
endstream
endobj
xref
0 6
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
0000000228 00000 n 
0000000305 00000 n 
trailer
<< /Size 6 /Root 1 0 R >>
startxref
1075
%%EOF
"""
    with open(SAMPLE_DIR / "product_workflow_api.pdf", "wb") as f:
        f.write(pdf_content)
    print("Generated sample_docs/product_workflow_api.pdf")


if __name__ == "__main__":
    generate_hr_policy_md()
    generate_security_guidelines_txt()
    generate_employee_handbook_docx()
    generate_product_workflow_api_pdf()
    print("All sample documents successfully generated in ./sample_docs!")
