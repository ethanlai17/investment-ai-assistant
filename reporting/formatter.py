from pathlib import Path

from jinja2 import Environment, FileSystemLoader
from loguru import logger

from models.types import ReportData


class ReportFormatter:
    def __init__(self, outputs_dir: str):
        self._outputs_dir = Path(outputs_dir)
        self._outputs_dir.mkdir(exist_ok=True)
        self._env = Environment(
            loader=FileSystemLoader(Path(__file__).parent / "templates"),
            trim_blocks=True,
            lstrip_blocks=True,
        )
        self._env.filters["zip"] = zip

    def render(
        self,
        report_data: ReportData,
        suffix: str = "",
        report_title: str = "Investment Report",
    ) -> tuple[Path, Path]:
        md_content = self._render_template("report.md.j2", report_data, report_title)
        txt_content = self._render_template("report.txt.j2", report_data, report_title)

        md_path = self._outputs_dir / f"{report_data.date}{suffix}.md"
        txt_path = self._outputs_dir / f"{report_data.date}{suffix}.txt"

        md_path.write_text(md_content, encoding="utf-8")
        txt_path.write_text(txt_content, encoding="utf-8")

        logger.info(f"Reports saved: {md_path}, {txt_path}")
        return md_path, txt_path

    def _render_template(
        self,
        template_name: str,
        report_data: ReportData,
        report_title: str = "Investment Report",
    ) -> str:
        template = self._env.get_template(template_name)
        return template.render(
            date=report_data.date,
            report_title=report_title,
            market_summary=report_data.market_summary,
            recommendations=report_data.recommendations,
            top_picks=report_data.top_picks,
            notes=report_data.notes,
        )
