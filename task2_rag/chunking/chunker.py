from pathlib import Path



def split_text(
        text: str,
        chunk_size: int = 1024,
        overlap: int = 128
):

    chunks = []

    start = 0

    text_length = len(text)


    while start < text_length:


        end = min(
            start + chunk_size,
            text_length
        )


        if end < text_length:

            split_pos = text.rfind(
                "\n",
                start,
                end
            )

            if split_pos > start:
                end = split_pos



        chunk = text[start:end].strip()


        if chunk:

            chunks.append(chunk)



        new_start = end - overlap


        if new_start <= start:

            new_start = end


        start = new_start



    return chunks



# 保留兼容
chunk_text = split_text



if __name__ == "__main__":


    file = Path("../parsed").glob("*.md")

    md = list(file)[0]


    text = md.read_text(
        encoding="utf-8"
    )


    chunks = split_text(text)


    print(
        "Chunk数量:",
        len(chunks)
    )