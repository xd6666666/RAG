
from pathlib import Path
import subprocess


TASK_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = TASK_DIR.parent

MINERU_EXECUTABLE = (
    PROJECT_DIR / ".venv-mineru" / "bin" / "mineru"
)

OUTPUT_DIR = TASK_DIR / "parsed"


def parse_pdf(pdf_path: Path) -> Path:
    """
    使用 MinerU 4.x 解析完整 PDF。
    返回解析后 Markdown 文件的路径。
    """
    pdf_path = Path(pdf_path).resolve()

    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF 不存在: {pdf_path}")

    if not MINERU_EXECUTABLE.is_file():
        raise FileNotFoundError(
            f"找不到 MinerU: {MINERU_EXECUTABLE}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_file = OUTPUT_DIR / f"{pdf_path.stem}.md"

    command = [
        str(MINERU_EXECUTABLE),
        "parse",
        str(pdf_path),
        "--pages", "all",
        "--format", "markdown",
        "--output", str(output_file),
        "--wait", "600",
        "--remote",
    ]

    print(f"[MinerU] 正在解析：{pdf_path.name}")

    # 不使用 --remote，要求使用本地解析能力
    subprocess.run(command, check=True)

    if not output_file.is_file():
        raise RuntimeError(
            f"MinerU 没有生成预期文件：{output_file}"
        )

    if output_file.stat().st_size == 0:
        raise RuntimeError("MinerU 输出了空文件")

    print(f"[MinerU] 解析完成：{output_file}")

    return output_file


if __name__ == "__main__":
    pdf_dir = TASK_DIR / "documents"
    pdf_files = sorted(pdf_dir.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"请将 PDF 放入：{pdf_dir}"
        )

    for pdf in pdf_files:
        parse_pdf(pdf)
