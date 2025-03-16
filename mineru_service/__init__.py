import asyncio
import base64
import gc
import logging
import os
import shutil
import tarfile
import tempfile
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, cast
from typing_extensions import override

import filetype
import fitz
import litserve as ls
import torch
from apscheduler.schedulers.background import BackgroundScheduler
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from magic_pdf.tools.cli import convert_file_to_pdf, do_parse
from starlette.datastructures import FormData, UploadFile

logger = logging.getLogger("uvicorn")

load_dotenv()

output_dir_base = os.getenv("MINERU_SERVICE_OUTPUT_DIR", "/tmp")

MINERU_SERVICE_OUTPUT_RETENTION_HOURS = int(os.getenv("MINERU_SERVICE_OUTPUT_RETENTION_HOURS", "6"))
MINERU_SERVICE_CLEANUP_INTERVAL_MINUTES = int(os.getenv("MINERU_SERVICE_CLEANUP_INTERVAL_MINUTES", "60"))


def cleanup_output_directory():
    output_dir = Path(output_dir_base) / "mineru"

    if not output_dir.exists():
        return

    logger.info(f"Start cleaning up output directory: {output_dir}")
    now = datetime.now()
    retention_time = now - timedelta(hours=MINERU_SERVICE_OUTPUT_RETENTION_HOURS)

    cnt = 0
    for item in output_dir.iterdir():
        try:
            mtime = datetime.fromtimestamp(item.stat().st_mtime)

            if mtime < retention_time:
                if item.is_dir():
                    shutil.rmtree(item, ignore_errors=True)
                else:
                    item.unlink()
                cnt += 1
        except Exception:
            pass

    logger.info(f"{cnt} output files has been cleaned up")


class API(ls.LitAPI):
    def __init__(self, output_dir="/tmp"):
        self.output_dir = Path(output_dir) / "mineru"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @override
    def setup(self, device):
        if device.startswith("cuda"):
            os.environ["CUDA_VISIBLE_DEVICES"] = device.split(":")[-1]
            if torch.cuda.device_count() > 1:
                raise RuntimeError("Remove any CUDA actions before setting 'CUDA_VISIBLE_DEVICES'.")

        from magic_pdf.model.doc_analyze_by_custom_model import ModelSingleton

        model_manager = ModelSingleton()
        model_manager.get_model(True, False)
        model_manager.get_model(False, False)
        logger.info(f"Model initialization complete on {device}!")

    def decode_request(self, request):
        request = cast(FormData, request)
        file = request.get("file")
        if isinstance(file, UploadFile):
            base64_encoded_file = base64.b64encode(asyncio.run(file.read())).decode("utf-8")
        elif isinstance(file, str):
            base64_encoded_file = file
        else:
            raise HTTPException(status_code=400, detail="Invalid file format")
        file_bytes = self.cvt2pdf(base64_encoded_file)
        opts = cast(dict[str, Any], request.get("kwargs", {}))
        opts.setdefault("debug_able", False)
        opts.setdefault("parse_method", "auto")
        return file_bytes, opts

    @override
    def predict(self, inputs):
        try:
            pdf_name = str(uuid.uuid4())
            output_dir = self.output_dir.joinpath(pdf_name)
            do_parse(self.output_dir, pdf_name, inputs[0], [], **inputs[1])
            tar_file = output_dir.with_suffix(".tar")
            with tarfile.open(tar_file, "w") as tar:
                tar.add(output_dir, arcname=pdf_name)
            return output_dir
        except Exception as e:
            shutil.rmtree(output_dir, ignore_errors=True)
            logger.exception(e)
            raise HTTPException(status_code=500, detail=str(e)) from None
        finally:
            self.clean_memory()

    @override
    def encode_response(self, response):
        return {
            "output_dir": response,
            "output_tarball_url": os.path.join(
                "/static",
                Path(response).with_suffix(".tar").relative_to(self.output_dir),
            ),
        }

    def clean_memory(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        gc.collect()

    @staticmethod
    def cvt2pdf(file_base64):
        try:
            temp_dir = Path(tempfile.mkdtemp())
            temp_file = temp_dir.joinpath("tmpfile")
            file_bytes = base64.b64decode(file_base64)
            file_ext = filetype.guess_extension(file_bytes)

            if file_ext in ["pdf", "jpg", "png", "doc", "docx", "ppt", "pptx"]:
                if file_ext == "pdf":
                    return file_bytes
                elif file_ext in ["jpg", "png"]:
                    with fitz.open(stream=file_bytes, filetype=file_ext) as f:
                        return f.convert_to_pdf()
                else:
                    temp_file.write_bytes(file_bytes)
                    convert_file_to_pdf(temp_file, temp_dir)
                    return temp_file.with_suffix(".pdf").read_bytes()
            else:
                raise Exception("Unsupported file format")
        except Exception as e:
            logger.exception(e)
            raise HTTPException(status_code=500, detail=str(e)) from None
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


def create_server():
    api = API(output_dir=output_dir_base)

    server = ls.LitServer(
        api,
        accelerator="cuda",
        devices=1,
        workers_per_device=1,
        timeout=False,
        track_requests=True,
    )

    server.app.mount("/static", StaticFiles(directory=os.path.join(output_dir_base, "mineru"), html=True))

    original_lifespan = server.app.router.lifespan_context

    @asynccontextmanager
    async def custom_lifespan(app: FastAPI):
        scheduler = BackgroundScheduler()
        scheduler.add_job(
            cleanup_output_directory,
            "interval",
            minutes=MINERU_SERVICE_CLEANUP_INTERVAL_MINUTES,
        )

        logger.info("Start scheduler")

        scheduler.start()
        async with original_lifespan(app):
            yield

        logger.info("Shutdown scheduler")
        scheduler.shutdown()

    server.app.router.lifespan_context = custom_lifespan

    return server


server = create_server()

if __name__ == "__main__":
    import fire

    fire.Fire(server.run)
