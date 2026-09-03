import fitz
import pymupdf4llm
import pathlib


def extract_pdf_to_txt(
    pdf_path,
    output_txt_path,
    start_page,
    end_page,
    header_margin=40,
    footer_margin=40
):
    print("deleting headers and footers:")

    page_numbers = list(range(start_page - 1, end_page))

    try:
        doc = fitz.open(pdf_path)
        for page_num in page_numbers:
            if page_num < len(doc):
                page = doc[page_num]
                rect = page.rect

                header_rect = fitz.Rect(
                    rect.x0,
                    rect.y0,
                    rect.x1,
                    rect.y0 + header_margin
                )

                footer_rect = fitz.Rect(
                    rect.x0,
                    rect.y1 - footer_margin,
                    rect.x1,
                    rect.y1
                )
                page.add_redact_annot(header_rect)
                page.add_redact_annot(footer_rect)
                page.apply_redactions()

        extracted_text = pymupdf4llm.to_markdown(
            doc,
            pages=page_numbers
        )

        output_file = pathlib.Path(output_txt_path)
        output_file.write_bytes(
            extracted_text.encode("utf-8")
        )

        print(
            f"done"
            f"saved in'{output_txt_path}'"
        )

    except Exception as e:
        print(f"ERROR: {e}")

    finally:
        if 'doc' in locals():
            doc.close()


if __name__ == "__main__":
    INPUT_PDF = "Questions Campbell.pdf"
    OUTPUT_TXT = "Questions Campbell.txt"

    START = 1
    END = 68

    extract_pdf_to_txt(
        INPUT_PDF,
        OUTPUT_TXT,
        START,
        END,
        header_margin=40,
        footer_margin=40
    )