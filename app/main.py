from funcs import sample
from CharRNN import CharRNN
import torch
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

app = FastAPI(docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/docs", include_in_schema=False)
async def swagger_ui_html(req: Request) -> HTMLResponse:
    root_path = req.scope.get("root_path", "").rstrip("/")
    openapi_url = root_path + app.openapi_url
    oauth2_redirect_url = app.swagger_ui_oauth2_redirect_url
    if oauth2_redirect_url:
        oauth2_redirect_url = root_path + oauth2_redirect_url
    return get_swagger_ui_html(
        openapi_url=openapi_url,
        title="PlatonovGen",
        oauth2_redirect_url=oauth2_redirect_url,
        init_oauth=app.swagger_ui_init_oauth,
        swagger_favicon_url="/static/favicon.ico",
        swagger_ui_parameters=app.swagger_ui_parameters,
    )


@app.get("/", include_in_schema=False)
async def root():
    return {"message": "Go to /docs to use the API"}


class TextGenerationRequest(BaseModel):
    prime: str = "Начало генерации"
    size: int = 2000


class TextGenerationResponse(BaseModel):
    generated_text: str


@app.post("/generate_text/", response_model=TextGenerationResponse,
          summary="Generate text from requested prime",
          description="Type the beginning of the text you want to generate "
                      "in the 'prime' field.")
async def generate_text(request: TextGenerationRequest) \
        -> TextGenerationResponse:
    if torch.cuda.is_available():
        map_location = torch.device('cuda')
    else:
        map_location = torch.device('cpu')
    with open("rnn.net", "rb") as f:
        checkpoint = torch.load(f, map_location=map_location)

    loaded = CharRNN(
        checkpoint["tokens"],
        n_hidden=checkpoint["n_hidden"],
        n_layers=checkpoint["n_layers"]
    )
    loaded.load_state_dict(checkpoint["state_dict"])

    generated_text = sample(loaded, request.size, top_k=5, prime=request.prime)

    return TextGenerationResponse(generated_text=generated_text)


@app.post("/generate_text_file/", response_model=TextGenerationResponse,
          summary="Generate text from uploaded file",
          description="Upload a text file with the beginning of the text you "
                      "want to generate. You can upload *.txt or *.csv files.")
def generate_text_file(file: UploadFile = File(...), size: int = 2000) \
        -> TextGenerationResponse:
    file_text = file.file.read().decode("utf-8")

    if torch.cuda.is_available():
        map_location = torch.device('cuda')
    else:
        map_location = torch.device('cpu')
    with open("rnn.net", "rb") as f:
        checkpoint = torch.load(f, map_location=map_location)

    loaded = CharRNN(
        checkpoint["tokens"],
        n_hidden=checkpoint["n_hidden"],
        n_layers=checkpoint["n_layers"]
    )
    loaded.load_state_dict(checkpoint["state_dict"])

    generated_text = sample(loaded, size, top_k=5, prime=file_text)

    return TextGenerationResponse(generated_text=generated_text)
