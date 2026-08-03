import io
import logging

try:
    from weasyprint import HTML, CSS   # tools to convert HTML into PDF
    WEASYPRINT_INSTALLED = True
except ImportError:
    WEASYPRINT_INSTALLED = False   # library missing, mark it unavailable instead of crashing

logger = logging.getLogger('ats_resume_scorer')   # logger for tracking issues in this module


"""
This function takes several HTML documents (like a resume report split into parts) and merges them into a single combined PDF file. It's used to produce one downloadable PDF from multiple HTML sections

"""

def generate_combined_pdf(html_docs: dict[str, str]) -> bytes:
    # stop early if the required library isn't installed
    if not WEASYPRINT_INSTALLED:
        raise ImportError("WeasyPrint is not installed. PDF generation unavailable.")

    documents = []   # will hold each HTML converted into a WeasyPrint document

    # convert each HTML string into a renderable document object
    for name, html_str in html_docs.items():
        doc = HTML(string=html_str).render()
        documents.append(doc)

    # use the first document as the base to merge everything into
    first_doc = documents[0]
    for other_doc in documents[1:]:   # loop through remaining documents
        for page in other_doc.pages:
            first_doc.pages.append(page)   # add each page into the first document

    # convert the merged document into final PDF bytes
    pdf_bytes = first_doc.write_pdf()
    return pdf_bytes

