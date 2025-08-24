from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

def build_pdf(title: str, content: str, output_path: str):
    doc = SimpleDocTemplate(output_path, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph(f"<b>{title}</b>", styles['Title']))
    story.append(Spacer(1, 20))

    for section in content.split("\n\n"):
        story.append(Paragraph(section.strip(), styles['BodyText']))
        story.append(Spacer(1, 12))

    doc.build(story)
