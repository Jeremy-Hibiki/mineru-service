import asyncio
import base64
import gc
import logging
import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any, cast
from typing_extensions import override

import filetype
import fitz
import litserve as ls
import torch
from dotenv import load_dotenv
from fastapi import HTTPException
from fastapi.staticfiles import StaticFiles
from starlette.datastructures import FormData, UploadFile

logger = logging.getLogger("uvicorn")

load_dotenv()


class API(ls.LitAPI):
    def __init__(self, output_dir="/tmp"):
        self.output_dir = Path(output_dir)

    @override
    def setup(self, device):
        if device.startswith("cuda"):
            os.environ["CUDA_VISIBLE_DEVICES"] = device.split(":")[-1]
            if torch.cuda.device_count() > 1:
                raise RuntimeError("Remove any CUDA actions before setting 'CUDA_VISIBLE_DEVICES'.")

        from magic_pdf.model.doc_analyze_by_custom_model import ModelSingleton
        from magic_pdf.tools.cli import convert_file_to_pdf, do_parse

        self.do_parse = do_parse
        self.convert_file_to_pdf = convert_file_to_pdf

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
            self.do_parse(self.output_dir, pdf_name, inputs[0], [], **inputs[1])
            return output_dir
        except Exception as e:
            shutil.rmtree(output_dir, ignore_errors=True)
            logger.exception(e)
            raise HTTPException(status_code=500, detail=str(e)) from None
        finally:
            self.clean_memory()

    @override
    def encode_response(self, response):
        return {"output_dir": response}

    def clean_memory(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        gc.collect()

    def cvt2pdf(self, file_base64):
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
                    self.convert_file_to_pdf(temp_file, temp_dir)
                    return temp_file.with_suffix(".pdf").read_bytes()
            else:
                raise Exception("Unsupported file format")
        except Exception as e:
            logger.exception(e)
            raise HTTPException(status_code=500, detail=str(e)) from None
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


api = API(output_dir="/tmp")


def create_server():
    server = ls.LitServer(
        api,
        accelerator="cuda",
        devices=1,
        workers_per_device=1,
        timeout=False,
        track_requests=True,
    )

    server.app.mount("/static", StaticFiles(directory="/tmp", html=True))
    return server


server = create_server()

if __name__ == "__main__":
    import fire

    fire.Fire(server.run)
