import json

with open("abfallentsorgung.json", "rb") as f:
    docs = json.load(f)


def format_information(item):
    lines = []
    lines.append(f"Informationen zu: {item['title']}")
    lines.append(f"url: {item['url']}")
    lines.append(f"Kurzbeschreibung: {item['description'] or 'Keine'}")
    lines.append("Sektionen:")

    for section in item["contents"]:
        # lines.append(f"Inhalt: {section['section_content']}")
        section_links = (
            "\n".join(
                f"{link['text']}: {link['link']}" for link in section["section_links"]
            )
            or "Keine"
        )
        lines.append(
            f"{section['section_title']}: {section['section_content']}\n Links:\n {section_links}\n"
        )

    return "\n".join(lines)


docs_txt = format_information(docs[0])
with open("abfallentsorgung.txt", "w") as f:
    f.write(docs_txt)
